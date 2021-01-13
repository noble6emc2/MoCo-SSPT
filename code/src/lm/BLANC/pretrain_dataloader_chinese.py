import sys
import os
import threading
import numpy as np
import torch
import json
import run_mrqa_blanc_pretraining_chinese as p_cn
from pytorch_pretrained_bert.tokenization import BasicTokenizer, BertTokenizer
from multiprocessing import Process, Queue


def move_to_cuda(sample):
    if len(sample) == 0:
        return {}

    def _move_to_cuda(maybe_tensor):
        if torch.is_tensor(maybe_tensor):
            return maybe_tensor.cuda()
        elif isinstance(maybe_tensor, dict):
            return {
                key: _move_to_cuda(value)
                for key, value in maybe_tensor.items()
            }
        elif isinstance(maybe_tensor, list):
            return [_move_to_cuda(x) for x in maybe_tensor]
        else:
            return maybe_tensor

    return _move_to_cuda(sample)

def tokenize(text):
    tokens = []
    for t in text:
        tokens.append(t)
    return tokens

def convert_tokens_to_ids(tokens, word2id):
    token_ids = []
    for idx, f in enumerate(tokens):
        token_ids.append(word2id[f] if f in word2id else word2id['[UNK]'])
    return token_ids

def get_training_data_queue(args):
    def enqueue(q, offset):
        print("train file offset: ", offset)
        fi = open(args.train_file, 'rb')
        cache = [None] * 10000
        first_time = True
        tokenizer = BertTokenizer.from_pretrained(
            args.tokenizer, do_lower_case=args.do_lower_case)

        while True:
            #print('first_time:', first_time)
            #sys.stdout.flush()
            
            if first_time:
                fi.seek(int(offset))
                first_time = False
            else:
                fi.seek(0)

            for line in fi:
                try:
                    line = line.rstrip().decode('utf-8')
                    sample_json = json.loads(line)
                except UnicodeDecodeError:
                    print(f"WARNING: one training line decode utf-8 ERROR")
                    sys.stdout.flush()
                    continue
                except json.decoder.JSONDecodeError:
                    print(f"WARNING: json.decoder.JSONDecodeError  ERROR")
                    sys.stdout.flush()
                    continue

                examples = p_cn.read_chinese_examples(
                    line_list=line, is_training=True)
                train_features = p_cn.convert_examples_to_features(
                    examples=examples,
                    tokenizer=tokenizer,
                    max_seq_length=args.max_seq_length,
                    doc_stride=args.doc_stride,
                    max_query_length=args.max_query_length,
                    is_training=True)

                if len(train_features) != 1:
                    print("get_training_data_data_queue WARNING: length of train_features != 1")

                '''
                if max([l for a, l in data[2]]) <= 0:
                    print(f"WARNING: {data[0]} has no positive label*********************")
                    sys.stdout.flush()
                    continue
                '''

                insert_idx = np.random.randint(0, len(cache))
                if cache[insert_idx] is None:
                    cache[insert_idx] = train_features[0]
                else:
                    q.put(cache[insert_idx])
                    cache[insert_idx] = train_features[0]

                del line
                del sample_json

    total_bytes = os.path.getsize(args.train_file)
    print("train file total bytes: ", total_bytes)

    if sys.version_info.major == 3:
        import queue
        q = queue.Queue(maxsize=500000)
    else:
        import Queue
        q = Queue.Queue(maxsize=500000)

    for i in range(args.enqueue_thread_num):
        print("enqueue thread started : ", i)
        enqeue_thread = threading.Thread(target=enqueue, args=(q, i * np.random.rand() * total_bytes / (args.enqueue_thread_num + 1)))
        enqeue_thread.setDaemon(True)
        enqeue_thread.start()
    return q


def multi_process_get_training_data_queue(args, start, end, p_list):
    def enqueue(q, offset):
        print("train file offset: ", offset)
        fi = open(args.train_file, 'rb')
        cache = [None] * 10000
        first_time = True
        tokenizer = BertTokenizer.from_pretrained(
            args.tokenizer, do_lower_case=args.do_lower_case)

        while True:
            #print('first_time:', first_time)
            #sys.stdout.flush()
            
            if first_time:
                fi.seek(int(offset))
                first_time = False
            elif start > end:
                fi.seek(0)
            else:
                fi.seek(int(start))

            for line in fi:
                if fi.tell() >= end:
                    break

                try:
                    line = line.rstrip().decode('utf-8')
                    sample_json = json.loads(line)
                except UnicodeDecodeError:
                    #print(f"WARNING: one training line decode utf-8 ERROR")
                    #print(line)
                    #sys.stdout.flush()
                    continue
                except json.decoder.JSONDecodeError as json_e:
                    #print(f"WARNING: json.decoder.JSONDecodeError  ERROR")
                    #print(line)
                    #print(json_e)
                    #sys.stdout.flush()
                    continue

                examples = p_cn.read_chinese_examples(
                    line_list=[line], is_training=True, 
                    first_answer_only=True, 
                    replace_mask="[unused1]",
                    do_lower_case=args.do_lower_case,
                    remove_query_in_passage=args.remove_query_in_passage)

                if len(examples) == 0:
                    continue
                
                train_features = p_cn.convert_chinese_examples_to_features(
                    examples=examples,
                    tokenizer=tokenizer,
                    max_seq_length=args.max_seq_length,
                    doc_stride=args.doc_stride,
                    max_query_length=args.max_query_length,
                    is_training=True,
                    first_answer_only=True)

                '''
                if len(train_features) != 1:
                    print("get_training_data_data_queue WARNING: length of train_features != 1")
                '''

                '''
                if max([l for a, l in data[2]]) <= 0:
                    print(f"WARNING: {data[0]} has no positive label*********************")
                    sys.stdout.flush()
                    continue
                '''

                for feature in train_features:
                    insert_idx = np.random.randint(0, len(cache))
                    if cache[insert_idx] is None:
                        cache[insert_idx] = feature
                    else:
                        q.put(cache[insert_idx])
                        cache[insert_idx] = feature

                del line
                del sample_json

    total_bytes = os.path.getsize(args.train_file)
    print("train file total bytes: ", total_bytes)

    q = Queue(maxsize=32767)

    for i in range(args.enqueue_thread_num):  # for fine tuning, thread num CAN be set 1.
        # offset = i * np.random.rand() * total_bytes / (args.enqueue_thread_num + 1)
        offset = np.random.rand() * (end - start) + start
        print("enqueue process started : ", i, offset, offset / total_bytes)
        p = Process(target=enqueue, args=(q, offset))
        p.start()
        p_list.append(p)

    return q


global_q_a = None
global_q_b = None
global_q = None


def get_training_batch_chinese(args, co_training: bool, p_list: list):
    total_bytes = os.path.getsize(args.train_file)
    global global_q_a
    global global_q_b
    split_byte = np.random.rand() * total_bytes
    q_a = multi_process_get_training_data_queue(args, 
        split_byte, (split_byte + total_bytes // 2) % total_bytes, p_list)
    q_b = multi_process_get_training_data_queue(args, 
        (split_byte + total_bytes // 2) % total_bytes, split_byte, p_list)
    global_q_a = q_a
    global_q_b = q_b
    feature_buffer = []
    batch_indicator = 0
    while True:
        new_feature_a = q_a.get()
        new_feature_b = q_b.get()
        feature_buffer.append((new_feature_a,
            new_feature_b))
        #print('after q.get')
        #sys.stdout.flush()
        batch_indicator += 1
        if batch_indicator == args.train_batch_size:  # ignore the reminders
            batch_input_ids = torch.tensor([f.input_ids for f, _ in feature_buffer], dtype=torch.long)
            batch_input_mask = torch.tensor([f.input_mask for f, _ in feature_buffer], dtype=torch.long)
            batch_segment_ids = torch.tensor([f.segment_ids for f, _ in feature_buffer], dtype=torch.long)
            batch_start_positions = torch.tensor([f.start_positions for f, _ in feature_buffer], dtype=torch.long)
            batch_end_positions = torch.tensor([f.end_positions for f, _ in feature_buffer], dtype=torch.long)
            #print("------------co-training--------------")
            #for feature, _ in feature_buffer:
            #    print(feature)
            #    break

            #print(len(feature_buffer))
            batch_a = batch_input_ids, batch_input_mask, batch_segment_ids, batch_start_positions, batch_end_positions

            batch_input_ids = torch.tensor([f.input_ids for _, f in feature_buffer], dtype=torch.long)
            batch_input_mask = torch.tensor([f.input_mask for _, f in feature_buffer], dtype=torch.long)
            batch_segment_ids = torch.tensor([f.segment_ids for _, f in feature_buffer], dtype=torch.long)
            batch_start_positions = torch.tensor([f.start_positions for _, f in feature_buffer], dtype=torch.long)
            batch_end_positions = torch.tensor([f.end_positions for _, f in feature_buffer], dtype=torch.long)
            #print("-------------co-training-------------")
            #for _, feature in feature_buffer:
            #    print(feature)
            #    break

            #print(len(feature_buffer))
            batch_b = batch_input_ids, batch_input_mask, batch_segment_ids, batch_start_positions, batch_end_positions

            if co_training:
                yield batch_a, batch_b
            else:
                yield batch_a
                yield batch_b
            
            batch_indicator = 0
            feature_buffer = []

        