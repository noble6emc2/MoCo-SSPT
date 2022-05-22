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
from pytorch_pretrained_bert.moco_dataloader import moco_dataloader
from torch.utils.data import DataLoader, TensorDataset
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.file_utils import WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.moco_bert import BLANC, BertForQuestionAnswering, BertMoCo
from pytorch_pretrained_bert.evaluation import MRQAEvaluator, SQuADEvaluator, output_loss
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
from pytorch_pretrained_bert.tokenization import BasicTokenizer, BertTokenizer, whitespace_tokenize
from pytorch_pretrained_bert.tokenization import _is_punctuation, _is_whitespace, _is_control
from mrqa_official_eval import exact_en_match_score, f1_en_score, metric_max_over_ground_truths
from pytorch_pretrained_bert.dataset_processor import MRQAProcessor, SQuADProcessor, CMRCProcessor, PretrainingProcessor

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
            if args.model_type == "BertForQA":
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
                    data_loader = moco_dataloader()
                    get_batch_fn = data_loader.get_training_moco_batch_english
                else:
                    raise NotImplementedError('This training language is not support')

                dataloader_iterator = enumerate(
                    get_batch_fn(args = args)
                    )

                for step, batch_dict in tqdm(
                    dataloader_iterator, 
                    total = args.num_iteration):
                    if step >= args.num_iteration:
                        data_loader.stop_iterate()
                        break

                    batch = batch_dict["sspt"]
                    if n_gpu == 1:
                        batch = tuple(t.to(device) for t in batch)

                    input_ids, input_mask, segment_ids, start_positions, end_positions = batch
                    #print(input_ids, input_ids.size())
                    #input()
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
                            #Save dataloader state
                            loader_state = data_loader.save_state()
                            #print(type(loader_state["offset_list"]))
                            output_dataloader_file = os.path.join(args.output_dir, 'dataloader_state.pkl')
                            with open(output_dataloader_file, "wb") as fout:
                                pkl.dump(loader_state, fout)
                            #Save training step & optimizerstate
                            optimizer_file = os.path.join(args.output_dir, 'training_num_optimzer_state.pkl')
                            torch.save({
                                "epoch": epoch,
                                "training_step": step + 1,
                                "optimizer": optimizer.state_dict()
                            }, optimizer_file)





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
    parser.add_argument('--test_mode', type=str, default='metrics')
    parser.add_argument('--enqueue_thread_num', type=int, default=4)
    parser.add_argument('--version_2_with_negative', type=bool, default=False)
    parser.add_argument('--warmup_dataloader', type=bool, default=False)
    parser.add_argument('--select_with_overall_losses', type=bool, default=False)
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--theta', type=float, default=0.3)
    parser.add_argument('--moving_loss_warmup_ratio', type=float, default=0.3)
    parser.add_argument('--moving_loss_num', type=int, default=8)
    parser.add_argument('--is_idx_mask', type=bool, default=False)
    parser.add_argument('--jacc_thres', type=float, default=0.2)
    parser.add_argument('--neg_drop_rate', type=float, default=0.4)
    parser.add_argument('--max_warmup_query_length', type=int, default=40)
    parser.add_argument('--max_comma_num', type=int, default=5)
    parser.add_argument('--warmup_window_size', type=int, default=8)
    args = parser.parse_args()

    if args.command == 'pretraining':
        logger.info('enter pretraining....')
        main(args)
    else:
        raise NotImplementedError("This command is not supported")
