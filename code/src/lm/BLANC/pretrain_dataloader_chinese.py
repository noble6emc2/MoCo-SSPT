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
            args.model, do_lower_case=args.do_lower_case)

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


def multi_process_get_training_data_queue(args):
    def enqueue(q, offset):
        print("train file offset: ", offset)
        fi = open(args.train_file, 'rb')
        cache = [None] * 10000
        first_time = True
        tokenizer = BertTokenizer.from_pretrained(
            args.model, do_lower_case=args.do_lower_case)

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
                    line_list=[line], is_training=True, first_answer_only=True)
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

    q = Queue(maxsize=500000)

    for i in range(args.enqueue_thread_num):  # for fine tuning, thread num CAN be set 1.
        # offset = i * np.random.rand() * total_bytes / (args.enqueue_thread_num + 1)
        offset = np.random.rand() * total_bytes
        print("enqueue process started : ", i, offset, offset / total_bytes)
        p = Process(target=enqueue, args=(q, offset))
        p.start()
    return q


global_q = None


def get_training_batch_chinese(args):
    global global_q
    q = multi_process_get_training_data_queue(args)
    global_q = q
    feature_buffer = []

    while True:
        new_feature = q.get()
        feature_buffer.append(new_feature)
        #print('after q.get')
        #sys.stdout.flush()
        batch_indicator += 1
        if batch_indicator == args.batch_size:  # ignore the reminders
            batch_input_ids = torch.tensor([f.input_ids for f in feature_buffer], dtype=torch.long)
            batch_input_mask = torch.tensor([f.input_mask for f in feature_buffer], dtype=torch.long)
            batch_segment_ids = torch.tensor([f.segment_ids for f in feature_buffer], dtype=torch.long)
            batch_start_positions = torch.tensor([f.start_position for f in feature_buffer], dtype=torch.long)
            batch_end_positions = torch.tensor([f.end_position for f in feature_buffer], dtype=torch.long)

            yield batch_input_ids, batch_input_mask, batch_segment_ids, batch_start_positions, batch_end_positions
            
            batch_indicator = 0
            feature_buffer = []

 
def load_eval_data_batches(args, word2id):
    re = []
    if args.eval_file == "":
        return re
    
    fi = open(args.eval_file, 'rb') 
    for line in fi:
        try:
            line = line.rstrip().decode('utf-8')
            sample_json = json.loads(line)
        except UnicodeDecodeError:
            print(f"load_eval_data WARNING: one training line decode utf-8 ERROR")
            sys.stdout.flush()
            continue
        except json.decoder.JSONDecodeError:
            print(f"load_eval_data WARNING: one training line decode utf-8 ERROR")
            sys.stdout.flush()
            continue
        
        data = convert_single_sample(sample_json, args, word2id, False, False)

        if args.eval_ans_recall==0:
            if sum([l for a, l in data[2]]) == 0:
                print(f"load_eval_data WARNING: {data[0]} has no positive label*********************")
                sys.stdout.flush()
                continue

        re.append(data)
        if len(re) % args.eval_max_sample_num == 0 and len(re)>0:
            yield re
            re = []
    yield re; return

def load_eval_data(args, word2id):
    # need cut off the max seq(query) len, adding [CLS] and [SEP] and keep the real len, no padding.
    if args.eval_file == "":
        return []

    re = []
    fi = open(args.eval_file, 'rb')
    for line in fi:
        try:
            line = line.rstrip().decode('utf-8')
            sample_json = json.loads(line)
        except UnicodeDecodeError:
            print(f"load_eval_data WARNING: one training line decode utf-8 ERROR")
            sys.stdout.flush()
            continue
        except json.decoder.JSONDecodeError:
            print(f"load_eval_data WARNING: one training line decode utf-8 ERROR")
            sys.stdout.flush()
            continue

        data = convert_single_sample(sample_json, args, word2id, False, False)

        if args.eval_ans_recall==0:
            if sum([l for a, l in data[2]]) == 0:
                print(f"load_eval_data WARNING: {data[0]} has no positive label*********************")
                sys.stdout.flush()
                continue

        re.append(data)
    print(f'eval data num: {len(re)}')
    return re

def debug_load_eval_data(args, word2id):
    re = []
    for i in range(5):
        qid = i
        q_len = np.random.randint(1, min(30, args.max_query_len))
        q_ids = np.random.randint(0, 10000, (q_len,))
        list_of_a_ids_and_label = []
        for i in range(np.random.randint(10, 100)):
            a_len = np.random.randint(100, min(300, args.max_seq_len))
            a_ids = torch.from_numpy(np.random.randint(0, 10000, (a_len,)))
            label = np.random.randint(0, 2)
            list_of_a_ids_and_label.append((a_ids, label))
        re.append((qid, torch.from_numpy(q_ids), list_of_a_ids_and_label))
    return re

