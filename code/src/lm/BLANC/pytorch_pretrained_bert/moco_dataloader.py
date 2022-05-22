import sys
import os
import threading
from more_itertools import first
import numpy as np
import torch
import json
import argparse
import pickle as pkl
import sys
import logging
import time
import random
sys.path.append('..')
#from tokenization import BasicTokenizer, BertTokenizer
#from dataset_processor import PretrainingProcessor
from pytorch_pretrained_bert.tokenization import BasicTokenizer, BertTokenizer
from pytorch_pretrained_bert.dataset_processor import PretrainingProcessor
from multiprocessing import Process, Queue, Manager

class moco_dataloader:
    def __init__(self,
            q_list = None, q_ptr = None, 
            s_e_list = None, offset_list = None
        ) -> None:
        self.data_processor = PretrainingProcessor()
        manager = Manager()
        self.share_state_dict = manager.dict()
        self.p_list = []
        def gen_q_list(q_list):
            ret_q_list = [Queue(maxsize=500) for _ in range(args.enqueue_thread_num)] #32767
            for i in range(len(q_list)):
                map(ret_q_list[i].put, q_list[i])

            return ret_q_list

        if q_list:
            self.q_list = gen_q_list(q_list)
        else:
            self.q_list = q_list

        self.q_ptr = q_ptr
        self.start_end_list = s_e_list 
        self.offset_list = offset_list
        self.logger = logging.getLogger(__name__)

    
    def stop_iterate(self):
        assert len(self.p_list) > 0
        for p in self.p_list:
            if p.is_alive():
                p.terminate()
                p.join()
    
    
    def save_state(self):
        def if_queue_full():
            for q in self.q_list:
                if not q.full():
                    return False

            return True

        def get_all_items(queue):
            ret_list = []
            while not queue.empty():
                ret_list.append(queue.get())

            return ret_list


        self.logger.info("waiting for queues to be full...")
        while not if_queue_full():
            time.sleep(2)
        
        self.logger.info("queues are full. continue to next step")
        #self.stop_iterate()
        print(self.share_state_dict)
        #print(self.q_ptr)
        #input()
        temp_dict = dict(self.share_state_dict)
        ret_dict ={
           "offset_list": [
               temp_dict[i] for i in range(len(temp_dict))
               ],
           "start_end_list": self.start_end_list,
           "q_list": [get_all_items(q) for q in self.q_list],
           "q_ptr": self.q_ptr 
        }
        return ret_dict


    @classmethod
    def load_state(cls, file_path):
        with open(file_path, "rb") as fin:
            state_dict = pkl.load(fin)

        return cls(
            q_list = state_dict['q_list'], 
            q_ptr = state_dict['q_ptr'],
            s_e_list = state_dict['start_end_list'], 
            offset_list = state_dict['offset_list'],
            )


    def get_features(self, tokenizer, args, line):
        examples = self.data_processor.read_english_examples(
            line_list=[line], is_training=True, 
            first_answer_only=True, 
            replace_mask="[unused1]",
            do_lower_case=args.do_lower_case)
        #self.logger.info(examples)

        if len(examples) == 0:
            return None
        
        train_features = self.data_processor.convert_english_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=True,
            first_answer_only=True,
            use_ag = False)

        if examples[0].ag_question_text is not None:
            ag_train_features = self.data_processor.convert_english_examples_to_features(
                examples=examples,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length,
                doc_stride=args.doc_stride,
                max_query_length=args.max_query_length,
                is_training=True,
                first_answer_only=True,
                use_ag = True)
        else:
            ag_train_features = [None for _ in range(len(train_features))]

        return train_features, ag_train_features

    def multi_process_get_moco_data_queue_en(self, args, start, end):
        def enqueue(q_list, offset, start_end_list, process_num, share_state_dict):
            #share_state_dict[process_num] = {
            #    "offset": None
            #}
            #share_state_dict[process_num] = "test"
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
                if chunked_start > chunked_end:
                    assert int(offset) >= chunked_start or int(offset) <= chunked_end
                else:
                    assert int(offset) >= chunked_start and int(offset) <= chunked_end

                if first_time:
                    first_time = False
                    fi.seek(int(offset))
                    if chunked_start > chunked_end:
                        if int(offset) <= chunked_end:
                            run_to_eof = False
                        elif int(offset) >= chunked_start:
                            run_to_eof = True
                        else:
                            raise NotImplementedError("This branch should not be reached")
                    else:
                        run_to_eof = False
                elif chunked_start > chunked_end:
                    if run_to_eof:
                        fi.seek(0)
                        run_to_eof = False
                    else:
                        fi.seek(int(chunked_start))
                        run_to_eof = True
                elif chunked_start <= chunked_end:
                    fi.seek(int(chunked_start))
                    run_to_eof = False

                #print("fi.tell", fi.tell())
                share_state_dict[process_num] = fi.tell()
                
                for line in fi:
                    if not run_to_eof and fi.tell() >= chunked_end:
                        break

                    try:
                        line = line.rstrip().decode('utf-8')
                        sample_json = json.loads(line)
                        sspt_json = json.loads(sample_json["sspt_line"])
                        moco_json = json.loads(sample_json["moco_line"])
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

                    ag_json = {**sspt_json}
                    if moco_json["question"] != "no corresponding answer":
                        ag_json["ag_question"] = moco_json["question"]

                    train_sspt_features, train_moco_features = self.get_features(
                        tokenizer, args, json.dumps(ag_json)
                        )
                    #if not train_sspt_features or not train_moco_features:
                    #    continue

                    for feature_pair in zip(train_sspt_features, train_moco_features):
                        insert_idx = np.random.randint(0, len(cache))
                        if cache[insert_idx] is None:
                            cache[insert_idx] = feature_pair
                        else:
                            q_list[process_num].put(cache[insert_idx])
                            cache[insert_idx] = feature_pair

                    del line
                    del sample_json
                    share_state_dict[process_num] = fi.tell()

                '''print('process_num', process_num)
                print("chunked_start, chunked_end", chunked_start, chunked_end)
                print("pointer", fi.tell())
                print('line', line)'''

        total_bytes = os.path.getsize(args.train_file)
        print("train file total bytes: ", total_bytes)

        if not self.q_list:
            self.q_list = [Queue(maxsize=500) for _ in range(args.enqueue_thread_num)] #32767
            chunk_size = ((end - start) // args.enqueue_thread_num if end >= start 
                else (end - start + total_bytes) // args.enqueue_thread_num)

        if not self.start_end_list:
            self.start_end_list = []
            for i in range(args.enqueue_thread_num):
                self.start_end_list.append(
                    ((start + chunk_size * i) % total_bytes, 
                    (start + chunk_size * (i + 1)) % total_bytes)
                )

        for i in range(args.enqueue_thread_num):  # for fine tuning, thread num CAN be set 1.
            # offset = i * np.random.rand() * total_bytes / (args.enqueue_thread_num + 1)
            chunked_start, chunked_end = self.start_end_list[i]
            #offset = chunked_start #np.random.rand() * (end - start) + start
            if not self.offset_list:
                offset = ((np.random.rand() * (chunked_end - chunked_start + total_bytes) + chunked_start) % total_bytes
                    if chunked_start > chunked_end else
                    (np.random.rand() * (chunked_end - chunked_start) + chunked_start) % total_bytes)
            else:
                offset = self.offset_list[i]

            print("enqueue process started : ", i, offset, offset / total_bytes)
            p = Process(
                target=enqueue,
                args=(self.q_list, offset, self.start_end_list, i, self.share_state_dict)
                )
            p.start()
            self.p_list.append(p)


    def get_training_moco_batch_english(self, args):
        total_bytes = os.path.getsize(args.train_file)
        split_byte = np.random.rand() * total_bytes
        self.multi_process_get_moco_data_queue_en(args, 
            split_byte, (split_byte + total_bytes // 2) % total_bytes)
        feature_buffer = {"sspt": [], "moco": []}
        batch_indicator = 0
        if not self.q_ptr:
            self.q_ptr = 0

        def get_feature_batch(feature_buffer):
            batch_input_ids = torch.tensor([f.input_ids for f in feature_buffer], dtype=torch.long)
            batch_input_mask = torch.tensor([f.input_mask for f in feature_buffer], dtype=torch.long)
            batch_segment_ids = torch.tensor([f.segment_ids for f in feature_buffer], dtype=torch.long)
            batch_start_positions = torch.tensor([f.start_positions for f in feature_buffer], dtype=torch.long)
            batch_end_positions = torch.tensor([f.end_positions for f in feature_buffer], dtype=torch.long)
            batch = (
                batch_input_ids, batch_input_mask, batch_segment_ids, 
                batch_start_positions, batch_end_positions
                )

            return batch

        while True:
            sspt_new_feature, moco_new_feature = self.q_list[self.q_ptr].get()
            feature_buffer["sspt"].append(sspt_new_feature)
            feature_buffer["moco"].append(moco_new_feature)
            #print('after q.get')
            #sys.stdout.flush()
            batch_indicator += 1
            self.q_ptr = (self.q_ptr + 1) % args.enqueue_thread_num
            #print('batch q_ptr', self.q_ptr)
            if batch_indicator == args.train_batch_size:  # ignore the reminders
                sspt_b = get_feature_batch(feature_buffer["sspt"])
                moco_indices = [
                    idx for idx in range(len(feature_buffer["moco"]))
                    if feature_buffer["moco"][idx]
                    ]
                moco_feature = [
                    f for f in feature_buffer["moco"] if f
                ]
                moco_b = get_feature_batch(moco_feature)
                yield {"sspt": sspt_b, "moco": moco_b, 'moco_indices': moco_indices}
                batch_indicator = 0
                feature_buffer = {"sspt": [], "moco": []}
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", default="/research/d4/gds/mindahu21/bert_model/bert-base-uncased-vocab.txt", type=str, required=False)
    parser.add_argument("--train_file", default="/research/d4/gds/mindahu21/ssptGen/000/sspt_qg_combined_dataset.jsonl", type=str)
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
    parser.add_argument('--new_cotraining_optimizer', type=bool, default=False)
    parser.add_argument('--is_idx_mask', type=bool, default=False)
    parser.add_argument('--jacc_thres', type=float, default=0.2)
    parser.add_argument('--neg_drop_rate', type=float, default=0.4)
    parser.add_argument('--max_warmup_query_length', type=int, default=40)
    parser.add_argument('--max_comma_num', type=int, default=5)
    parser.add_argument('--warmup_window_size', type=int, default=8)
    args = parser.parse_args()
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    root.addHandler(handler)

    cnt_iter = 0

    
    '''data_loader = moco_dataloader()
    for batch in data_loader.get_training_moco_batch_english(args = args):
        if cnt_iter % 2 == 0:
            import pickle as pkl
            loader_state = data_loader.save_state()
            print(type(loader_state["offset_list"]))
            with open("test_dataloader_state_dict.pkl", "wb") as fout:
                pkl.dump(loader_state, fout)

        if cnt_iter >= 10:
            data_loader.stop_iterate()
            break

        cnt_iter += 1
        print(batch)
        input()'''

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    data_loader = moco_dataloader.load_state("test_dataloader_state_dict.pkl")
    for batch in data_loader.get_training_moco_batch_english(args = args):
        if cnt_iter >= 1:
            data_loader.stop_iterate()
            break

        cnt_iter += 1
        print(batch)
        input()