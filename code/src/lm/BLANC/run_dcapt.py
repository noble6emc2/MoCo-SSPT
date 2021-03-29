# Copyright (c) 2019, Facebook, Inc. and its affiliates. All Rights Reserved
"""Run BERT on MRQA."""

from __future__ import absolute_import, division, print_function

from tqdm import tqdm
import pickle as pkl
import argparse
import collections
import json
import logging
import math
import os
import random
import time
import sys
import gzip
import six
import unicodedata
import string
import re
import math
from torch import nn
from io import open
sys.path.append(os.path.dirname(__file__))

import numpy as np
import torch
import datetime
import pytorch_pretrained_bert.pretrain_dataloader as dataloader
from torch.utils.data import DataLoader, TensorDataset
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.file_utils import WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.blanc import BLANC, BertForQuestionAnswering
from pytorch_pretrained_bert.evaluation import MRQAEvaluator, SQuADEvaluator
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
from pytorch_pretrained_bert.tokenization import BasicTokenizer, BertTokenizer, whitespace_tokenize
from pytorch_pretrained_bert.tokenization import _is_punctuation, _is_whitespace, _is_control
from mrqa_official_eval import exact_en_match_score, f1_en_score, metric_max_over_ground_truths
from pytorch_pretrained_bert.dataset_processor import MRQAProcessor, SQuADProcessor, CMRCProcessor

PRED_FILE = "predictions.json"
EVAL_FILE = "eval_results.txt"
TEST_FILE = "test_results.txt"
MODEL_TYPES = ["BLANC", "BertForQA"]
DATASET_TYPES = ["MRQA", "SQuAD", "CMRC"]
LANG_TYPES = ['EN', 'CN']
COMMANDS = ['finetuning', 'pretraining', 'cotraining', 'testing']

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
    logger.info("device: {}, n_gpu: {}, 16-bits training: {}".format(
        device, n_gpu, args.fp16))

    from torch.utils.tensorboard import SummaryWriter
    # default `log_dir` is "runs" - we'll be more specific here
    writer = SummaryWriter(
        os.path.join(args.output_dir, "baseline_training_loss_lmb_%s/" % str(args.lmb) 
        + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))
    args.train_batch_size = \
        args.train_batch_size // args.gradient_accumulation_steps

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if args.do_train:
        assert (args.train_file is not None) and (args.dev_file is not None)

    if args.eval_test:
        assert args.test_file is not None
    else:
        assert args.dev_file is not None

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if args.do_train:
        logger.addHandler(logging.FileHandler(os.path.join(args.output_dir, "train.log"), 'w'))
    else:
        logger.addHandler(logging.FileHandler(os.path.join(args.output_dir, "eval.log"), 'w'))
    logger.info(args)

    tokenizer = BertTokenizer.from_pretrained(
        args.tokenizer, do_lower_case=args.do_lower_case)

    if args.do_train:
        num_train_optimization_steps = \
            args.num_iteration // args.gradient_accumulation_steps * args.num_train_epochs
        logger.info("***** Train *****")
        logger.info("  Num orig examples = %d", args.num_iteration * args.train_batch_size)
        logger.info("  Num split examples = %d", args.num_iteration)
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        eval_step = max(1, args.num_iteration // args.eval_per_epoch)
        best_result = None
        lrs = [args.learning_rate] if args.learning_rate else [1e-6, 2e-6, 3e-6, 5e-6, 1e-5, 2e-5, 3e-5, 5e-5]
        for lr in lrs:
            if args.model_type == "BLANC":
                model, pretrained_weights = BLANC.from_pretrained(
                    args.model, cache_dir=PYTORCH_PRETRAINED_BERT_CACHE)
            elif args.model_type == "BertForQA":
                model, pretrained_weights = BertForQuestionAnswering.from_pretrained(
                    args.model, cache_dir=PYTORCH_PRETRAINED_BERT_CACHE)
            else:
                raise NotImplementedError("Unknown Model Type")

            if args.fp16:
                model.half()
                
            model.to(device)
            if n_gpu > 1:
                model = torch.nn.DataParallel(model)
            param_optimizer = list(model.named_parameters())
            param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer
                            if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                {'params': [p for n, p in param_optimizer
                            if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]

            if args.fp16:
                try:
                    from apex.optimizers import FP16_Optimizer
                    from apex.optimizers import FusedAdam
                except ImportError:
                    raise ImportError("Please install apex from https://www.github.com/nvidia/apex"
                                      "to use distributed and fp16 training.")
                optimizer = FusedAdam(optimizer_grouped_parameters,
                                      lr=lr,
                                      bias_correction=False,
                                      max_grad_norm=1.0)
                if args.loss_scale == 0:
                    optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
                else:
                    optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
            else:
                optimizer = BertAdam(optimizer_grouped_parameters,
                                     lr=lr,
                                     warmup=args.warmup_proportion,
                                     t_total=num_train_optimization_steps)
            global_step = 0
            start_time = time.time()
            from tqdm import tqdm
            p_list = []
            for epoch in range(int(args.num_train_epochs)):
                model.train()
                logger.info("Start epoch #{} (lr = {})...".format(epoch, lr))
                running_loss = 0.0
                if args.training_lang == 'EN':
                    get_batch_fn = dataloader.get_training_batch_english
                elif args.training_lang == 'CN':
                    get_batch_fn = dataloader.get_training_batch_chinese
                else:
                    raise NotImplementedError('This training language is not support')

                for step, batch in tqdm(enumerate(
                    get_batch_fn(
                        args, co_training = False, p_list = p_list)), 
                        total = args.num_iteration):
                    if step >= args.num_iteration:
                        for p in p_list:
                            if p.is_alive:
                                p.terminate()
                                p.join()

                        break

                    if n_gpu == 1:
                        batch = tuple(t.to(device) for t in batch)

                    input_ids, input_mask, segment_ids, start_positions, end_positions = batch
                    loss, _, _ = model(input_ids, segment_ids, input_mask, start_positions, end_positions, geometric_p=args.geometric_p, window_size=args.window_size, lmb=args.lmb)
                    #print("step", step,"loss", loss)

                    if n_gpu > 1:
                        loss = loss.mean()
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps
                        
                    if args.fp16:
                        optimizer.backward(loss)
                    else:
                        loss.backward()
                    
                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        if args.fp16:
                            lr_this_step = lr * \
                                warmup_linear(global_step/num_train_optimization_steps, args.warmup_proportion)
                            for param_group in optimizer.param_groups:
                                param_group['lr'] = lr_this_step
                        optimizer.step()
                        optimizer.zero_grad()
                        global_step += 1

                    running_loss += loss.item()
                    if (step + 1) % 500 == 0:
                       writer.add_scalar('Training loss of baseline with lmb %s' % str(args.lmb),
                            running_loss / 500,
                            epoch * args.num_iteration + step + 1)
                       running_loss = 0.0

                    if (step + 1) % eval_step == 0:
                        logger.info('Epoch: {}, Step: {} / {}, used_time = {:.2f}s'.format(
                            epoch, step + 1, args.num_iteration, time.time() - start_time))

                        save_model = False
                        if args.do_eval:
                            raise NotImplementedError('This branch should not be entered')
                        else:
                            save_model = True

                        if save_model:
                            model_to_save = model.module if hasattr(model, 'module') else model
                            output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
                            output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
                            torch.save(model_to_save.state_dict(), output_model_file)
                            model_to_save.config.to_json_file(output_config_file)
                            tokenizer.save_vocabulary(args.output_dir)
                            if best_result:
                                with open(os.path.join(args.output_dir, EVAL_FILE), "w") as writer:
                                    for key in sorted(best_result.keys()):
                                        writer.write("%s = %s\n" % (key, str(best_result[key])))


def main_cotraining(args):
    from torch.utils.tensorboard import SummaryWriter
    # default `log_dir` is "runs" - we'll be more specific here
    if args.co_training_mode == 'moving_loss':
        writer = SummaryWriter(os.path.join(args.output_dir_a, "co_training_moving_loss_num_%d/" % args.moving_loss_num + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
    else:
        writer = SummaryWriter(os.path.join(args.output_dir_a, "co_training_data_cur_theta_%s/" % str(args.theta) + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
    logger.info("device: {}, n_gpu: {}, 16-bits training: {}".format(
        device, n_gpu, args.fp16))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))
    args.train_batch_size = \
        args.train_batch_size // args.gradient_accumulation_steps

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if args.do_train:
        assert (args.train_file is not None) and (args.dev_file is not None)

    if args.eval_test:
        assert args.test_file is not None
    else:
        assert args.dev_file is not None

    if not os.path.exists(args.output_dir_a):
        #os.makedirs(args.output_dir)
        os.makedirs(args.output_dir_a)

    if not os.path.exists(args.output_dir_b):
        os.makedirs(args.output_dir_b)

    if args.do_train:
        logger.addHandler(logging.FileHandler(os.path.join(args.output_dir_a, "train.log"), 'w'))
    else:
        logger.addHandler(logging.FileHandler(os.path.join(args.output_dir_a, "eval.log"), 'w'))
    
    logger.info(args)
    tokenizer = BertTokenizer.from_pretrained(
        args.tokenizer, do_lower_case=args.do_lower_case)

    if args.do_train:
        num_train_optimization_steps = \
            args.num_iteration // args.gradient_accumulation_steps * args.num_train_epochs
        logger.info("***** Train *****")
        logger.info("  Num orig examples = %d", args.num_iteration * args.train_batch_size)
        logger.info("  Num split examples = %d", args.num_iteration)
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        eval_step = max(1, args.num_iteration // args.eval_per_epoch)
        best_result = None
        result_a = None
        result_b = None
        lrs = [args.learning_rate] if args.learning_rate else [1e-6, 2e-6, 3e-6, 5e-6, 1e-5, 2e-5, 3e-5, 5e-5]
        
        def create_optimizer(model_a, model_b, num_train_optimization_steps):
            param_optimizer_a = list(model_a.named_parameters())
            param_optimizer_b = list(model_b.named_parameters())
            param_optimizer_a = [n for n in param_optimizer_a if 'pooler' not in n[0]]
            param_optimizer_b = [n for n in param_optimizer_b if 'pooler' not in n[0]]
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters_a = [
                {'params': [p for n, p in param_optimizer_a
                            if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                {'params': [p for n, p in param_optimizer_a
                            if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            optimizer_grouped_parameters_b = [
                {'params': [p for n, p in param_optimizer_b
                            if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                {'params': [p for n, p in param_optimizer_b
                            if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]

            if args.fp16:
                try:
                    from apex.optimizers import FP16_Optimizer
                    from apex.optimizers import FusedAdam
                except ImportError:
                    raise ImportError("Please install apex from https://www.github.com/nvidia/apex"
                                      "to use distributed and fp16 training.")
                optimizer_a = FusedAdam(optimizer_grouped_parameters_a,
                                      lr=lr,
                                      bias_correction=False,
                                      max_grad_norm=1.0)
                optimizer_b = FusedAdam(optimizer_grouped_parameters_b,
                                      lr=lr,
                                      bias_correction=False,
                                      max_grad_norm=1.0)
                if args.loss_scale == 0:
                    optimizer_a = FP16_Optimizer(optimizer_a, dynamic_loss_scale=True)
                    optimizer_b = FP16_Optimizer(optimizer_b, dynamic_loss_scale=True)
                else:
                    optimizer_a = FP16_Optimizer(optimizer_a, static_loss_scale=args.loss_scale)
                    optimizer_b = FP16_Optimizer(optimizer_b, static_loss_scale=args.loss_scale)
            else:
                optimizer_a = BertAdam(optimizer_grouped_parameters_a,
                                     lr=lr,
                                     warmup=args.warmup_proportion,
                                     t_total=num_train_optimization_steps)
                optimizer_b = BertAdam(optimizer_grouped_parameters_b,
                                     lr=lr,
                                     warmup=args.warmup_proportion,
                                     t_total=num_train_optimization_steps)

            return optimizer_a, optimizer_b


        for lr in lrs:
            assert args.model_type == "BLANC"
            model_a, pretrained_weights_a = BLANC.from_pretrained(
                args.model, cache_dir=PYTORCH_PRETRAINED_BERT_CACHE)
            model_b, pretrained_weights_b = BLANC.from_pretrained(
                args.model, cache_dir=PYTORCH_PRETRAINED_BERT_CACHE)
            if args.fp16:
                model_a.half()
                model_b.half()

            model_a.to(device)
            model_b.to(device)
            if n_gpu > 1:
                model_a = torch.nn.DataParallel(model_a)
                model_b = torch.nn.DataParallel(model_b)

            max_warmup_step = int(
                num_train_optimization_steps * 
                args.moving_loss_warmup_ratio
                )
            if args.new_cotraining_optimizer:
                logger.info('creating warmup optimizer for pretraining...')
                optimizer_a, optimizer_b = create_optimizer(model_a, 
                    model_b, 
                    max_warmup_step)
            else:
                optimizer_a, optimizer_b = create_optimizer(model_a, 
                    model_b, num_train_optimization_steps)

            global_step = 0
            start_time = time.time()
            lmb_window_list_a = []
            lmb_window_list_b = []
            from tqdm import tqdm
            p_list = []
            first_in_cotraining = True
            for epoch in range(int(args.num_train_epochs)):
                logger.info("Start epoch #{} (lr = {})...".format(epoch, lr))
                running_loss_a = 0.0
                running_loss_b = 0.0
                if args.training_lang == 'EN':
                    get_batch_fn = dataloader.get_training_batch_english
                elif args.training_lang == 'CN':
                    get_batch_fn = dataloader.get_training_batch_chinese
                else:
                    raise NotImplementedError('This training language is not support')

                for step, (batch_a, batch_b) in tqdm(enumerate(
                    get_batch_fn(
                        args, co_training = True, p_list = p_list)),
                        total = args.num_iteration):
                    if step >= args.num_iteration:
                        for p in p_list:
                            if p.is_alive:
                                p.terminate()
                                p.join()

                        break

                    if n_gpu == 1:
                        batch_a = tuple(t.to(device) for t in batch_a)
                        batch_b = tuple(t.to(device) for t in batch_b)

                    step_ratio = global_step / num_train_optimization_steps
                    #Warm up in order to make Model A/B's hypothesis different
                    input_ids_a, input_mask_a, segment_ids_a, start_positions_a, end_positions_a = batch_a
                    input_ids_b, input_mask_b, segment_ids_b, start_positions_b, end_positions_b = batch_b
                    if global_step < max_warmup_step:
                        # warming up stage
                        model_a.train()
                        model_b.train()
                        
                        loss_a, _, _ = model_a(input_ids_a, segment_ids_a, input_mask_a, start_positions_a, end_positions_a, geometric_p=args.geometric_p, window_size=args.window_size, lmb=args.lmb)
                        loss_b, _, _ = model_b(input_ids_b, segment_ids_b, input_mask_b, start_positions_b, end_positions_b, geometric_p=args.geometric_p, window_size=args.window_size, lmb=args.lmb)
                    else:
                        # co-training stage
                        if args.new_cotraining_optimizer and first_in_cotraining:
                            logger.info("creating new optimizer for cotraining...")
                            optimizer_a, optimizer_b = create_optimizer(model_a, 
                                model_b,
                                num_train_optimization_steps - global_step
                                )
                            first_in_cotraining = False
                            
                        model_a.eval()
                        model_b.eval()

                        with torch.no_grad():
                            _, _, context_losses_a = model_a(input_ids_b, segment_ids_b, input_mask_b, start_positions_b, end_positions_b, geometric_p=args.geometric_p, window_size=args.window_size, lmb=args.lmb)
                            _, _, context_losses_b = model_b(input_ids_a, segment_ids_a, input_mask_a, start_positions_a, end_positions_a, geometric_p=args.geometric_p, window_size=args.window_size, lmb=args.lmb)

                            lmb_list_a = [i for i in context_losses_b.detach().cpu().numpy()]
                            lmb_list_b = [i for i in context_losses_a.detach().cpu().numpy()]

                            if args.debug:
                                print("context_losses_a", context_losses_a, "context_losses_b", context_losses_b)
                                print('lmb_list_a', lmb_list_a, 'lmb_list_b', lmb_list_b)

                        model_a.train()
                        model_b.train()

                        if args.co_training_mode == 'moving_loss':
                            if len(lmb_window_list_a) + args.train_batch_size > args.moving_loss_num:
                                pop_num = abs(args.moving_loss_num - len(lmb_window_list_a) - args.train_batch_size)
                                lmb_window_list_a = (lmb_window_list_a[pop_num:] + lmb_list_a)
                                lmb_window_list_b = (lmb_window_list_b[pop_num:] + lmb_list_b)
                            else:
                                lmb_window_list_a += lmb_list_a
                                lmb_window_list_b += lmb_list_b
                            
                            moving_loss_a = np.mean(lmb_window_list_a)
                            moving_loss_b = np.mean(lmb_window_list_b)

                            lmbs_a = torch.tensor([args.lmb if l <= moving_loss_a else 0. for l in lmb_list_a])
                            lmbs_b = torch.tensor([args.lmb if l <= moving_loss_b else 0. for l in lmb_list_b])
                            mask_a = torch.tensor([1 if l <= moving_loss_a else 0 for idx, l in enumerate(lmb_list_a)])
                            mask_b = torch.tensor([1 if l <= moving_loss_b else 0 for idx, l in enumerate(lmb_list_b)])
                            if n_gpu == 1:
                                lmbs_a = lmbs_a.to(device)
                                lmbs_b = lmbs_b.to(device)
                                mask_a = mask_a.to(device)
                                mask_b = mask_b.to(device)

                            if args.debug:
                                print("lmb_window_list_a", lmb_window_list_a, "lmb_window_list_b", lmb_window_list_b)
                                print("moving_loss_a", moving_loss_a, "moving_loss_b", moving_loss_b)
                                print("lmbs_a", lmbs_a, "lmbs_b", lmbs_b)
                                print("mask_a", mask_a, "mask_b", mask_b)
                                input()

                            if args.is_idx_mask:
                                loss_a, _, _ = model_a(input_ids_a, segment_ids_a, input_mask_a, start_positions_a, end_positions_a, lmbs=None, geometric_p=args.geometric_p, window_size=args.window_size, lmb=args.lmb, batch_idx_mask=mask_a)
                                loss_b, _, _ = model_b(input_ids_b, segment_ids_b, input_mask_b, start_positions_b, end_positions_b, lmbs=None, geometric_p=args.geometric_p, window_size=args.window_size, lmb=args.lmb, batch_idx_mask=mask_b)
                                loss_a = torch.sum(loss_a) / torch.sum(mask_a)
                                loss_b = torch.sum(loss_b) / torch.sum(mask_b)
                            else:
                                loss_a, _, _ = model_a(input_ids_a, segment_ids_a, input_mask_a, start_positions_a, end_positions_a, lmbs=lmbs_a, geometric_p=args.geometric_p, window_size=args.window_size, lmb=args.lmb, batch_idx_mask=None,
                                context_attention_mask = mask_a)
                                loss_b, _, _ = model_b(input_ids_b, segment_ids_b, input_mask_b, start_positions_b, end_positions_b, lmbs=lmbs_b, geometric_p=args.geometric_p, window_size=args.window_size, lmb=args.lmb, batch_idx_mask=None,
                                context_attention_mask = mask_b)

                        elif args.co_training_mode == 'data_cur':
                            top_k_index_a = set(np.argsort(lmb_list_a)[:math.ceil(args.theta * len(lmb_list_a))])
                            top_k_index_b = set(np.argsort(lmb_list_b)[:math.ceil(args.theta * len(lmb_list_b))])

                            lmbs_a = torch.tensor([args.lmb if idx in top_k_index_a else 0.
                                                    for idx in range(len(lmb_list_a))])
                            lmbs_b = torch.tensor([args.lmb if idx in top_k_index_b else 0.
                                                    for idx in range(len(lmb_list_b))])
                            mask_a = torch.tensor([1 if idx in top_k_index_a else 0
                                                    for idx in range(len(lmb_list_a))])
                            mask_b = torch.tensor([1 if idx in top_k_index_b else 0
                                                    for idx in range(len(lmb_list_b))])
                            if n_gpu == 1:
                                lmbs_a = lmbs_a.to(device)
                                lmbs_b = lmbs_b.to(device)
                                mask_a = mask_a.to(device)
                                mask_b = mask_b.to(device)
                                            
                            if args.debug:
                                print("top_k_index_a", top_k_index_a, "top_k_index_b", top_k_index_b)
                                print("lmbs_a", lmbs_a, "lmbs_b", lmbs_b)
                                print("mask_a", mask_a, "mask_b", mask_b)
                                input()

                            if args.is_idx_mask:
                                loss_a, _, _ = model_a(input_ids_a, segment_ids_a, input_mask_a, start_positions_a, end_positions_a, lmbs=None, geometric_p=args.geometric_p, window_size=args.window_size, lmb=args.lmb, batch_idx_mask=mask_a)
                                loss_b, _, _ = model_b(input_ids_b, segment_ids_b, input_mask_b, start_positions_b, end_positions_b, lmbs=None, geometric_p=args.geometric_p, window_size=args.window_size, lmb=args.lmb, batch_idx_mask=mask_b)
                                loss_a = torch.sum(loss_a) / torch.sum(mask_a)
                                loss_b = torch.sum(loss_b) / torch.sum(mask_b)
                            else:
                                loss_a, _, _ = model_a(input_ids_a, segment_ids_a, input_mask_a, start_positions_a, end_positions_a, lmbs=lmbs_a, geometric_p=args.geometric_p, window_size=args.window_size, lmb=args.lmb, batch_idx_mask=None)
                                loss_b, _, _ = model_b(input_ids_b, segment_ids_b, input_mask_b, start_positions_b, end_positions_b, lmbs=lmbs_b, geometric_p=args.geometric_p, window_size=args.window_size, lmb=args.lmb, batch_idx_mask=None)
                        else:
                            raise Exception("Unsuppoted co training mode.")
                    
                    if n_gpu > 1:
                        loss_a = loss_a.mean()
                        loss_b = loss_b.mean()

                    if args.gradient_accumulation_steps > 1:
                        loss_a = loss_a / args.gradient_accumulation_steps
                        loss_b = loss_b / args.gradient_accumulation_steps

                    if args.fp16:
                        optimizer_a.backward(loss_a)
                        optimizer_b.backward(loss_b)
                    else:
                        loss_a.backward()
                        loss_b.backward()
                    
                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        if args.fp16:
                            lr_this_step = lr * \
                                warmup_linear(global_step/num_train_optimization_steps, args.warmup_proportion)
                            for param_group in optimizer_a.param_groups:
                                param_group['lr'] = lr_this_step
                            for param_group in optimizer_b.param_groups:
                                param_group['lr'] = lr_this_step

                        optimizer_a.step()
                        optimizer_a.zero_grad()
                        optimizer_b.step()
                        optimizer_b.zero_grad()
                        global_step += 1
 
                    running_loss_a += loss_a.item()
                    running_loss_b += loss_b.item()
                    if (step + 1) % 500 == 0:
                       writer.add_scalar('Training loss(Model A)',
                            running_loss_a / 500,
                            epoch * args.num_iteration + step + 1)
                       writer.add_scalar('Training loss(Model B)',
                            running_loss_b / 500,
                            epoch * args.num_iteration + step + 1)

                       running_loss_a = 0.0
                       running_loss_b = 0.0

                    if (step + 1) % eval_step == 0:
                        logger.info('Epoch: {}, Step: {} / {}, used_time = {:.2f}s'.format(
                            epoch, step + 1, args.num_iteration, time.time() - start_time))

                        save_model = False
                        if args.do_eval:
                            raise NotImplementedError('This branch should not be entered')
                        else:
                            save_model = True

                        if save_model:
                            model_to_save = model_a.module if hasattr(model_a, 'module') else model_a
                            output_dir_a_iter = args.output_dir_a + '_iter_%d' % step
                            output_dir_b_iter = args.output_dir_b + '_iter_%d' % step
                            if not os.path.exists(output_dir_a_iter):
                                os.makedirs(output_dir_a_iter)

                            if not os.path.exists(output_dir_b_iter):
                                os.makedirs(output_dir_b_iter)

                            output_model_file = os.path.join(output_dir_a_iter, WEIGHTS_NAME)
                            output_config_file = os.path.join(output_dir_a_iter, CONFIG_NAME)
                            torch.save(model_to_save.state_dict(), output_model_file)
                            model_to_save.config.to_json_file(output_config_file)
                            tokenizer.save_vocabulary(output_dir_a_iter)
                            if result_a:
                                with open(os.path.join(output_dir_a_iter, EVAL_FILE), "w") as writer:
                                    for key in sorted(result_a.keys()):
                                        writer.write("%s = %s\n" % (key, str(result_a[key])))

                            model_to_save = model_b.module if hasattr(model_b, 'module') else model_b
                            output_model_file = os.path.join(output_dir_b_iter, WEIGHTS_NAME)
                            output_config_file = os.path.join(output_dir_b_iter, CONFIG_NAME)
                            torch.save(model_to_save.state_dict(), output_model_file)
                            model_to_save.config.to_json_file(output_config_file)
                            tokenizer.save_vocabulary(output_dir_b_iter)
                            if result_b:
                                with open(os.path.join(output_dir_b_iter, EVAL_FILE), "w") as writer:
                                    for key in sorted(result_b.keys()):
                                        writer.write("%s = %s\n" % (key, str(result_b[key])))


def main_finetuning(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
    logger.info("device: {}, n_gpu: {}, 16-bits training: {}".format(
        device, n_gpu, args.fp16))

    from torch.utils.tensorboard import SummaryWriter
    # default `log_dir` is "runs" - we'll be more specific here
    tb_writer = SummaryWriter(
        os.path.join(args.output_dir, "fintuning_loss_lmb_%s/" % str(args.lmb) 
        + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))

    #writer.add_scalar('Training loss of finetuning with lmb test',0,1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))
    args.train_batch_size = \
        args.train_batch_size // args.gradient_accumulation_steps

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if args.do_train:
        assert (args.train_file is not None) and (args.dev_file is not None)

    if args.eval_test:
        assert args.test_file is not None
    else:
        assert args.dev_file is not None

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.do_train:
        logger.addHandler(logging.FileHandler(os.path.join(args.output_dir, "train.log"), 'w'))
    else:
        logger.addHandler(logging.FileHandler(os.path.join(args.output_dir, "eval.log"), 'w'))
    
    logger.info(args)
    tokenizer = BertTokenizer.from_pretrained(
        args.tokenizer, do_lower_case=args.do_lower_case)
    if args.do_eval and (args.do_train or (not args.eval_test)):
        if args.dataset_type == "SQuAD":
            with open(args.dev_file) as f:
                dataset_json = json.load(f)

            eval_dataset = dataset_json['data']
            eval_examples = SQuADProcessor.read_squad_examples(
                input_file=args.dev_file, is_training=False,
                version_2_with_negative=args.version_2_with_negative)
            eval_features = SQuADProcessor.convert_english_examples_to_features(
                examples=eval_examples,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length,
                doc_stride=args.doc_stride,
                max_query_length=args.max_query_length,
                is_training=False)
        elif args.dataset_type == "MRQA":
            with gzip.GzipFile(args.dev_file, 'r') as reader:
                content = reader.read().decode('utf-8').strip().split('\n')[1:]
                eval_dataset = [json.loads(line) for line in content]

            eval_examples = MRQAProcessor.read_mrqa_examples(
                    args.dev_file, is_training=True, 
                    first_answer_only=True, 
                    do_lower_case=True,
                    remove_query_in_passage=False)
            eval_features = MRQAProcessor.convert_english_examples_to_features(
                    examples=eval_examples,
                    tokenizer=tokenizer,
                    max_seq_length=args.max_seq_length,
                    doc_stride=args.doc_stride,
                    max_query_length=args.max_query_length,
                    is_training=True,
                    first_answer_only=True)
        elif args.dataset_type == 'CMRC':
            eval_examples, eval_dataset = CMRCProcessor.read_cmrc_examples(
                    args.dev_file, is_training=True, 
                    first_answer_only=True, 
                    do_lower_case=True,
                    remove_query_in_passage=False)
            eval_features = CMRCProcessor.convert_chinese_examples_to_features(
                        examples=eval_examples,
                        tokenizer=tokenizer,
                        max_seq_length=args.max_seq_length,
                        doc_stride=args.doc_stride,
                        max_query_length=args.max_query_length,
                        is_training=True,
                        first_answer_only=True)
        else:
            raise NotImplementedError("This dataset type is not supported")

        logger.info("***** Dev *****")
        logger.info("  Num orig examples = %d", len(eval_examples))
        logger.info("  Num split examples = %d", len(eval_features))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_example_index)
        eval_dataloader = DataLoader(eval_data, batch_size=args.eval_batch_size)

    if args.do_train:
        if args.dataset_type == "MRQA":
            train_examples = MRQAProcessor.read_mrqa_examples(
                    args.train_file, is_training=True, 
                    first_answer_only=True, 
                    do_lower_case=True,
                    remove_query_in_passage=False)
            train_features = MRQAProcessor.convert_english_examples_to_features(
                    examples=train_examples,
                    tokenizer=tokenizer,
                    max_seq_length=args.max_seq_length,
                    doc_stride=args.doc_stride,
                    max_query_length=args.max_query_length,
                    is_training=True,
                    first_answer_only=True)
        elif args.dataset_type == "SQuAD":
            train_examples = SQuADProcessor.read_squad_examples(
                    input_file=args.train_file, is_training=True, version_2_with_negative=args.version_2_with_negative)
            train_features = SQuADProcessor.convert_english_examples_to_features(
                    examples=train_examples,
                    tokenizer=tokenizer,
                    max_seq_length=args.max_seq_length,
                    doc_stride=args.doc_stride,
                    max_query_length=args.max_query_length,
                    is_training=True)
        elif args.dataset_type == "CRMC":
            train_examples, _ = CMRCProcessor.read_cmrc_examples(
                    args.train_file, is_training=True, 
                    first_answer_only=True, 
                    do_lower_case=True,
                    remove_query_in_passage=False)
            train_features = CMRCProcessor.convert_chinese_examples_to_features(
                    examples=train_examples,
                    tokenizer=tokenizer,
                    max_seq_length=args.max_seq_length,
                    doc_stride=args.doc_stride,
                    max_query_length=args.max_query_length,
                    is_training=True,
                    first_answer_only=True)
        else:
            raise NotImplementedError("This dataset type is not supported")

        if args.train_mode == 'sorted' or args.train_mode == 'random_sorted':
            train_features = sorted(train_features, key=lambda f: np.sum(f.input_mask))
        else:
            random.shuffle(train_features)

        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_start_positions = torch.tensor([f.start_position for f in train_features], dtype=torch.long)
        all_end_positions = torch.tensor([f.end_position for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                   all_start_positions, all_end_positions)
        train_dataloader = DataLoader(train_data, batch_size=args.train_batch_size)
        train_batches = [batch for batch in train_dataloader]

        num_train_optimization_steps = \
            len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        logger.info("***** Train *****")
        logger.info("  Num orig examples = %d", len(train_examples))
        logger.info("  Num split examples = %d", len(train_dataloader))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        eval_step = max(1, len(train_batches) // args.eval_per_epoch)
        best_result = None
        lrs = [args.learning_rate] if args.learning_rate else [1e-6, 2e-6, 3e-6, 5e-6, 1e-5, 2e-5, 3e-5, 5e-5]
        for lr in lrs:
            assert args.model_type in MODEL_TYPES
            model, pretrained_weights = BLANC.from_pretrained(
                args.model, cache_dir=PYTORCH_PRETRAINED_BERT_CACHE)
            if args.fp16:
                model.half()
            model.to(device)
            if n_gpu > 1:
                model = torch.nn.DataParallel(model)
            param_optimizer = list(model.named_parameters())
            param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer
                            if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                {'params': [p for n, p in param_optimizer
                            if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]

            if args.fp16:
                try:
                    from apex.optimizers import FP16_Optimizer
                    from apex.optimizers import FusedAdam
                except ImportError:
                    raise ImportError("Please install apex from https://www.github.com/nvidia/apex"
                                      "to use distributed and fp16 training.")
                optimizer = FusedAdam(optimizer_grouped_parameters,
                                      lr=lr,
                                      bias_correction=False,
                                      max_grad_norm=1.0)
                if args.loss_scale == 0:
                    optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
                else:
                    optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
            else:
                optimizer = BertAdam(optimizer_grouped_parameters,
                                     lr=lr,
                                     warmup=args.warmup_proportion,
                                     t_total=num_train_optimization_steps)
                                     
            global_step = 0
            start_time = time.time()
            from tqdm import tqdm
            for epoch in range(int(args.num_train_epochs)):
                model.train()
                logger.info("Start epoch #{} (lr = {})...".format(epoch, lr))
                running_loss = 0.0
                for step, batch in tqdm(enumerate(train_batches), total = len(train_batches)):
                    if n_gpu == 1:
                        batch = tuple(t.to(device) for t in batch)

                    input_ids, input_mask, segment_ids, start_positions, end_positions = batch
                    loss, _, _ = model(input_ids, segment_ids, input_mask, start_positions, end_positions, geometric_p=args.geometric_p, window_size=args.window_size, lmb=args.lmb)
                    #print("step", step,"loss", loss)

                    if n_gpu > 1:
                        loss = loss.mean()
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps
                        
                    if args.fp16:
                        optimizer.backward(loss)
                    else:
                        loss.backward()
                    
                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        if args.fp16:
                            lr_this_step = lr * \
                                warmup_linear(global_step/num_train_optimization_steps, args.warmup_proportion)
                            for param_group in optimizer.param_groups:
                                param_group['lr'] = lr_this_step
                        optimizer.step()
                        optimizer.zero_grad()
                        global_step += 1

                    running_loss += loss.item()
                    if (step + 1) % 500 == 0:
                       tb_writer.add_scalar('Training loss of finetuning with lmb %s' % str(args.lmb),
                            running_loss / 500,
                            epoch * len(train_batches) + step + 1)
                       running_loss = 0.0

                    if (step + 1) % eval_step == 0 or args.debug:
                        logger.info('Epoch: {}, Step: {} / {}, used_time = {:.2f}s'.format(
                            epoch, step + 1, len(train_batches), time.time() - start_time))

                        save_model = False
                        if args.do_eval:
                            if args.dataset_type == "MRQA":
                                result, _, _ = \
                                    MRQAEvaluator.evaluate(args, model, device, eval_dataset,
                                            eval_dataloader, eval_examples, eval_features)
                            elif args.dataset_type == "SQuAD":
                                result, _, _ = \
                                    SQuADEvaluator.evaluate(args, model, device, eval_dataset,
                                            eval_dataloader, eval_examples, eval_features)
                            elif args.dataset_type == "CMRC":
                                result, _, _ = \
                                    MRQAEvaluator.evaluate(args, model, device, eval_dataset,
                                            eval_dataloader, eval_examples, eval_features)
                            else:
                                raise NotImplementedError("Dataset type is not supported.")

                            model.train()
                            for res_k, res_v in result.items():
                                tb_writer.add_scalar(
                                    '%s in finetuning with lmb %s in dev set' % (res_k, str(args.lmb)),
                                    res_v,
                                    epoch * len(train_batches) + step + 1)

                            result['global_step'] = global_step
                            result['epoch'] = epoch
                            result['learning_rate'] = lr
                            result['batch_size'] = args.train_batch_size
                            print(result)

                            if (best_result is None) or (result[args.eval_metric] > best_result[args.eval_metric]):
                                best_result = result
                                save_model = True
                                logger.info("!!! Best dev %s (lr=%s, epoch=%d): %.2f" %
                                            (args.eval_metric, str(lr), epoch, result[args.eval_metric]))
                        else:
                            save_model = True

                        if save_model:
                            model_to_save = model.module if hasattr(model, 'module') else model
                            output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
                            output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
                            torch.save(model_to_save.state_dict(), output_model_file)
                            model_to_save.config.to_json_file(output_config_file)
                            tokenizer.save_vocabulary(args.output_dir)
                            if best_result:
                                with open(os.path.join(args.output_dir, EVAL_FILE), "w") as writer:
                                    for key in sorted(best_result.keys()):
                                        writer.write("%s = %s\n" % (key, str(best_result[key])))


def main_model_testing(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
    logger.info("device: {}, n_gpu: {}, 16-bits training: {}".format(
        device, n_gpu, args.fp16))
    tokenizer = BertTokenizer.from_pretrained(
        args.tokenizer, do_lower_case=args.do_lower_case)

    if args.model_type == "BertForQA":
        model, pretrained_weights = BertForQuestionAnswering.from_pretrained(
                    args.model, cache_dir=PYTORCH_PRETRAINED_BERT_CACHE)
    else:
        raise NotImplementedError("This model type is not supported")

    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    
    with open(args.dev_file) as f:
        dataset_json = json.load(f)

    eval_dataset = dataset_json['data']
    if args.dataset_type == "SQuAD":
        eval_examples = SQuADProcessor.read_squad_examples(
                input_file=args.dev_file, is_training=False,
                version_2_with_negative=args.version_2_with_negative
                )
        eval_features = SQuADProcessor.convert_squad_examples_to_features(
                examples=eval_examples,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length,
                doc_stride=args.doc_stride,
                max_query_length=args.max_query_length,
                is_training=False
                )
    else:
        raise NotImplementedError("This dataset type is not supported")

    logger.info("***** Dev *****")
    logger.info("  Num orig examples = %d", len(eval_examples))
    logger.info("  Num split examples = %d", len(eval_features))
    logger.info("  Batch size = %d", args.eval_batch_size)
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_example_index)
    eval_dataloader = DataLoader(eval_data, batch_size=args.eval_batch_size)
    if args.dataset_type == "SQuAD":
        result, _, _ = \
                SQuADEvaluator.evaluate(args, model, device, eval_dataset,
                            eval_dataloader, eval_examples, eval_features)
    else:
        raise NotImplementedError("Dataset type is not supported.")

    #model.train()
    result['batch_size'] = args.train_batch_size
    logger.info(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="bert-base-chinese", type=str, required=True)
    parser.add_argument("--model_type", choices=MODEL_TYPES, type=str, required=True)
    parser.add_argument("--dataset_type", choices=DATASET_TYPES, type=str, default='MRQA')
    parser.add_argument("--training_lang", choices=LANG_TYPES, type=str, required=True)
    parser.add_argument("--command", choices=COMMANDS, type=str, required=True)
    parser.add_argument("--tokenizer", default="bert-base-chinese", type=str, required=True)
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")
    parser.add_argument("--train_file", default=None, type=str)
    parser.add_argument("--dev_file", default=None, type=str)
    parser.add_argument("--test_file", default=None, type=str)
    parser.add_argument("--eval_per_epoch", default=10, type=int,
                        help="How many times it evaluates on dev set per epoch")
    parser.add_argument("--max_seq_length", default=384, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                                "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--doc_stride", default=128, type=int,
                        help="When splitting up a long document into chunks, "
                                "how much stride to take between chunks.")
    parser.add_argument("--max_query_length", default=64, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                                "be truncated to this length.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--train_mode", type=str, default='random', choices=['random', 'sorted', 'random_sorted'])
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
    #parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--eval_test", action="store_true", help="Whether to evaluate on final test set.")
    parser.add_argument("--train_batch_size", default=32, type=int, help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int, help="Total batch size for predictions.")
    parser.add_argument("--learning_rate", default=None, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=1.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--eval_metric", default='f1', type=str)
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% "
                                "of training.")
    parser.add_argument("--n_best_size", default=20, type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions.json "
                                "output file.")
    parser.add_argument("--max_answer_length", default=30, type=int,
                        help="The maximum length of an answer that can be generated. "
                                "This is needed because the start "
                                "and end predictions are not conditioned on one another.")
    parser.add_argument("--verbose_logging", action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                                "A number of warnings are expected for a normal MRQA evaluation.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                                "0 (default value): dynamic loss scaling.\n"
                                "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--geometric_p', type=float, default=0.3)
    parser.add_argument('--window_size', type=int, default=5)
    parser.add_argument('--num_iteration', type=int, default=9375)
    parser.add_argument('--lmb', type=float, default=0.5)
    parser.add_argument('--do_lower_case', type=bool, default=True)
    parser.add_argument('--remove_query_in_passage', type=bool, default=True)
    parser.add_argument('--co_training_mode', type=str, default='data_cur')
    parser.add_argument('--enqueue_thread_num', type=int, default=4)
    parser.add_argument('--version_2_with_negative', type=bool, default=False)
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--theta', type=float, default=0.8)
    parser.add_argument('--moving_loss_warmup_ratio', type=float, default=0.3)
    parser.add_argument('--moving_loss_num', type=int, default=8)
    parser.add_argument('--new_cotraining_optimizer', type=bool, default=False)
    parser.add_argument('--is_idx_mask', type=bool, default=False)
    args = parser.parse_args()
    args.output_dir_a = args.output_dir + "_a"
    args.output_dir_b = args.output_dir + "_b"

    print('------new_cotraining_optimizer-----: %s' % args.new_cotraining_optimizer)

    if args.command == 'cotraining':
        logger.info('enter cotraining....')
        main_cotraining(args)
    elif args.command == 'finetuning':
        logger.info('enter finetuning....')
        main_finetuning(args)
    elif args.command == 'testing':
        logger.info('enter testing....')
        main_model_testing(args)
    elif args.command == 'pretraining':
        logger.info('enter pretraining....')
        main(args)
    else:
        raise NotImplementedError("This command is not supported")
