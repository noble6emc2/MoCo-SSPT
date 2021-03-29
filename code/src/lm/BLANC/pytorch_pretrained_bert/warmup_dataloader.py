import sys
import os
import threading
import numpy as np
import torch
import json
import jieba
import unicodedata
import jieba.posseg as pseg
from pytorch_pretrained_bert.tokenization import BasicTokenizer, BertTokenizer
from pytorch_pretrained_bert.tokenization import _is_punctuation, _is_whitespace, _is_control
from pytorch_pretrained_bert.dataset_processor import PretrainingProcessor, MRQAExample
from multiprocessing import Process, Queue
jieba.enable_paddle()



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

def multi_process_get_warmup_data_queue_cn(args, start, end, p_list):
    def warmup_sample_filter(examples, do_lower_case = True, warmup_window_size = 5):

        def is_whitespace(c):
            if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
                return True

            cat = unicodedata.category(c)
            if cat == "Zs":
                return True
                
            return False

        assert len(examples) == 1
        for example in examples:
            if do_lower_case:
                raw_doc_tokens = list(jieba.cut(example.paragraph_text.lower()))
            else:
                raw_doc_tokens = list(jieba.cut(example.paragraph_text))
            doc_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True
            
            k = 0
            temp_word = ""
            for c in example.paragraph_text:
                temp_word += c
                char_to_word_offset.append(k)

                if temp_word == raw_doc_tokens[k]:
                    doc_tokens.append(temp_word)
                    temp_word = ""
                    k += 1

            if k != len(raw_doc_tokens):
                print("Warning: paragraph '{}' tokenization error ".format(example.paragraph_text))
                return False

            assert len(example.start_positions) == 1 and len(example.start_positions) == 1
            start_tok_position = char_to_word_offset[example.start_positions[0]]
            end_tok_position = char_to_word_offset[example.end_positions[0]]
            left_bound = max(start_tok_position - warmup_window_size, 0)
            right_bound = max(end_tok_position + warmup_window_size, len(raw_doc_tokens))
            context_tok_list = (
                [tok for tok in 
                    raw_doc_tokens[left_bound: start_tok_position]] + 
                [tok for tok in 
                    raw_doc_tokens[end_tok_position + 1 : right_bound]]
                )
            context_tok_set = set([tok
                for tok in context_tok_list if not (_is_punctuation(tok) or \
                    _is_whitespace(tok) or _is_control(tok))])

            if do_lower_case:
                question_tokens = list(jieba.cut(example.question_text.lower()))
            else:
                question_tokens = list(jieba.cut(example.question_text))

            question_tok_set = set([tok
                for tok in question_tokens if not (_is_punctuation(tok) or \
                    _is_whitespace(tok) or _is_control(tok))])

            jaccard = float(len(context_tok_set.intersection(question_tok_set)) 
                    / len(context_tok_set.union(question_tok_set)))

            

            

    def enqueue(q_list, offset, start_end_list, process_num):
        print("train file offset: ", offset)
        fi = open(args.train_file, 'rb')
        cache = [None] * 10000
        first_time = True
        tokenizer = BertTokenizer.from_pretrained(
            args.tokenizer, do_lower_case=args.do_lower_case)
        chunked_start, chunked_end = start_end_list[process_num]
        print("chunked_start, chunked_end", chunked_start, chunked_end)
        run_to_eof = False
        while True:
            if first_time:
                fi.seek(int(offset))
                first_time = False
                if chunked_start > chunked_end:
                    run_to_eof = True
            elif run_to_eof and chunked_start > chunked_end:
                fi.seek(0)
                run_to_eof = False
            else:
                fi.seek(int(chunked_start))
                if chunked_start > chunked_end:
                    run_to_eof = True

                '''q_list[process_num].put({
                        'feature': None, 
                        'example': None,
                        'finished': True
                        })
                return'''

            
            for line in fi:
                if not run_to_eof and fi.tell() >= chunked_end:
                    break

                try:
                    line = line.rstrip().decode('utf-8')
                    sample_json = json.loads(line)
                except UnicodeDecodeError:
                    '''print(f"WARNING: one training line decode utf-8 ERROR")
                    print(line)
                    sys.stdout.flush()'''
                    continue
                except json.decoder.JSONDecodeError as json_e:
                    '''print(f"WARNING: json.decoder.JSONDecodeError  ERROR")
                    print(line)
                    print(json_e)
                    sys.stdout.flush()'''
                    continue

                data_processor = PretrainingProcessor()
                examples =  data_processor.read_chinese_examples(
                    line_list=[line], is_training=True, 
                    first_answer_only=True, 
                    replace_mask="[unused1]",
                    do_lower_case=args.do_lower_case,
                    remove_query_in_passage=args.remove_query_in_passage)

                if len(examples) == 0:
                    continue
                
                train_features = data_processor.convert_chinese_examples_to_features(
                    examples=examples,
                    tokenizer=tokenizer,
                    max_seq_length=args.max_seq_length,
                    doc_stride=args.doc_stride,
                    max_query_length=args.max_query_length,
                    is_training=True,
                    first_answer_only=True)


                for feature in train_features:
                    insert_idx = np.random.randint(0, len(cache))
                    if cache[insert_idx] is None:
                        cache[insert_idx] = {
                            'feature': feature, 
                            'example': sample_json,
                            'finished': False
                        }
                    else:
                        q_list[process_num].put(cache[insert_idx])
                        cache[insert_idx] = {
                            'feature': feature, 
                            'example': sample_json,
                            'finished': False
                        }
                    '''q_list[process_num].put()'''

                del line
                del sample_json

    total_bytes = os.path.getsize(args.train_file)
    print("train file total bytes: ", total_bytes)

    q_list = [Queue(maxsize=32767) for _ in range(args.enqueue_thread_num)]
    chunk_size = ((end - start) // args.enqueue_thread_num if end >= start 
        else (end - start + total_bytes) // args.enqueue_thread_num) 
    start_end_list = []
    for i in range(args.enqueue_thread_num):
        start_end_list.append(
            ((start + chunk_size * i) % total_bytes, 
            (start + chunk_size * (i + 1)) % total_bytes)
        )

    for i in range(args.enqueue_thread_num):  # for fine tuning, thread num CAN be set 1.
        # offset = i * np.random.rand() * total_bytes / (args.enqueue_thread_num + 1)
        chunked_start, chunked_end = start_end_list[i]
        #chunked_start = 0
        #chunked_end = total_bytes
        #offset = chunked_start #np.random.rand() * (end - start) + start
        offset = ((np.random.rand() * (chunked_end - chunked_start + total_bytes) + chunked_start) % total_bytes
            if chunked_start > chunked_end else
            (np.random.rand() * (chunked_end - chunked_start) + chunked_start) % total_bytes)
        print("enqueue process started : ", i, offset, offset / total_bytes)
        p = Process(target=enqueue, args=(q_list, offset, start_end_list, i))
        p.start()
        p_list.append(p)

    return q_list


def multi_process_get_warmup_data_queue_en(args, start, end, p_list):
    def enqueue(q_list, offset, start_end_list, process_num):
        print("train file offset: ", offset)
        fi = open(args.train_file, 'rb')
        cache = [None] * 10000
        first_time = True
        tokenizer = BertTokenizer.from_pretrained(
            args.tokenizer, do_lower_case=args.do_lower_case)
        chunked_start, chunked_end = start_end_list[process_num]
        #print("chunked_start, chunked_end", chunked_start, chunked_end)
        run_to_eof = False
        while True:
            #print('first_time:', first_time)
            #sys.stdout.flush()
            if first_time:
                fi.seek(int(offset))
                first_time = False
                if chunked_start > chunked_end:
                    run_to_eof = True
            elif run_to_eof and chunked_start > chunked_end:
                fi.seek(0)
                #print("chunked_start, chunked_end, ptr", 
                #    chunked_start, chunked_end, fi.tell())
                run_to_eof = False
            else:
                fi.seek(int(chunked_start))
                if chunked_start > chunked_end:
                    run_to_eof = True

            
            for line in fi:
                if not run_to_eof and fi.tell() >= chunked_end:
                    break

                try:
                    line = line.rstrip().decode('utf-8')
                    sample_json = json.loads(line)
                except UnicodeDecodeError:
                    '''print(f"WARNING: one training line decode utf-8 ERROR")
                    print(line)
                    sys.stdout.flush()'''
                    continue
                except json.decoder.JSONDecodeError as json_e:
                    '''print(f"WARNING: json.decoder.JSONDecodeError  ERROR")
                    print(line)
                    print(json_e)
                    sys.stdout.flush()'''
                    continue

                #print('line', line)
                '''examples = p_cn.read_chinese_examples(
                    line_list=[line], is_training=True, 
                    first_answer_only=True, 
                    replace_mask="[unused1]",
                    do_lower_case=args.do_lower_case,
                    remove_query_in_passage=args.remove_query_in_passage)'''
                data_processor = PretrainingProcessor()
                examples =  data_processor.read_english_examples(
                    line_list=[line], is_training=True, 
                    first_answer_only=True, 
                    replace_mask="[unused1]",
                    do_lower_case=args.do_lower_case)

                if len(examples) == 0:
                    continue
                
                '''train_features = p_cn.convert_chinese_examples_to_features(
                    examples=examples,
                    tokenizer=tokenizer,
                    max_seq_length=args.max_seq_length,
                    doc_stride=args.doc_stride,
                    max_query_length=args.max_query_length,
                    is_training=True,
                    first_answer_only=True)'''
                train_features = data_processor.convert_english_examples_to_features(
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
                        q_list[process_num].put(cache[insert_idx])
                        cache[insert_idx] = feature

                del line
                del sample_json

            '''print('process_num', process_num)
            print("chunked_start, chunked_end", chunked_start, chunked_end)
            print("pointer", fi.tell())
            print('line', line)'''

    total_bytes = os.path.getsize(args.train_file)
    print("train file total bytes: ", total_bytes)

    q_list = [Queue(maxsize=32767) for _ in range(args.enqueue_thread_num)]
    chunk_size = ((end - start) // args.enqueue_thread_num if end >= start 
        else (end - start + total_bytes) // args.enqueue_thread_num) 
    start_end_list = []
    for i in range(args.enqueue_thread_num):
        start_end_list.append(
            ((start + chunk_size * i) % total_bytes, 
            (start + chunk_size * (i + 1)) % total_bytes)
        )

    for i in range(args.enqueue_thread_num):  # for fine tuning, thread num CAN be set 1.
        # offset = i * np.random.rand() * total_bytes / (args.enqueue_thread_num + 1)
        chunked_start, chunked_end = start_end_list[i]
        #offset = chunked_start #np.random.rand() * (end - start) + start
        offset = ((np.random.rand() * (chunked_end - chunked_start + total_bytes) + chunked_start) % total_bytes
            if chunked_start > chunked_end else
            (np.random.rand() * (chunked_end - chunked_start) + chunked_start) % total_bytes)
        print("enqueue process started : ", i, offset, offset / total_bytes)
        p = Process(target=enqueue, args=(q_list, offset, start_end_list, i))
        p.start()
        p_list.append(p)

    return q_list


global_q_a_cn_list = None
global_q_b_cn_list = None
global_q_a_en_list = None
global_q_b_en_list = None
global_q = None


def get_training_batch_chinese(args, co_training: bool, p_list: list):
    total_bytes = os.path.getsize(args.train_file)
    global global_q_a_cn_list
    global global_q_b_cn_list
    split_byte = np.random.rand() * total_bytes
    q_a_list = multi_process_get_training_data_queue_cn(args, 
        split_byte, (split_byte + total_bytes // 2) % total_bytes, p_list)
    q_b_list = multi_process_get_training_data_queue_cn(args, 
        (split_byte + total_bytes // 2) % total_bytes, split_byte, p_list)
    global_q_a_cn_list = q_a_list
    global_q_b_cn_list = q_b_list
    feature_buffer = []
    batch_indicator = 0
    q_ptr = 0
    while True:
        new_feature_a = q_a_list[q_ptr].get()
        new_feature_b = q_b_list[q_ptr].get()
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
        
        q_ptr = (q_ptr + 1) % args.enqueue_thread_num

def get_training_batch_english(args, co_training: bool, p_list: list):
    total_bytes = os.path.getsize(args.train_file)
    global global_q_a_en_list
    global global_q_b_en_list
    split_byte = np.random.rand() * total_bytes
    q_a_list = multi_process_get_training_data_queue_en(args, 
        split_byte, (split_byte + total_bytes // 2) % total_bytes, p_list)
    q_b_list = multi_process_get_training_data_queue_en(args, 
        (split_byte + total_bytes // 2) % total_bytes, split_byte, p_list)
    global_q_a_en_list = q_a_list
    global_q_b_en_list = q_b_list
    feature_buffer = []
    batch_indicator = 0
    q_ptr = 0
    while True:
        new_feature_a = q_a_list[q_ptr].get()
        new_feature_b = q_b_list[q_ptr].get()
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
        
        q_ptr = (q_ptr + 1) % args.enqueue_thread_num

        