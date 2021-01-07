import run_mrqa_blanc_pretraining_chinese as p_cn
from pytorch_pretrained_bert.tokenization import BasicTokenizer, BertTokenizer
import pretrain_dataloader_chinese as cn_dataloader
import argparse
if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained(
            "bert-base-chinese")

    print(p_cn.tokenize_chinese("1903年，威廉·哈雷（WilliamHarley）和戴维森（Davidson）两兄弟在 [UNK] 密尔沃基创立了著名的哈雷戴维森品牌。", masking = "[UNK]", do_lower_case = True))
    print(tokenizer.tokenize("[UNK]"))
    with open('test_train.txt', 'r') as fin:
        for line in fin:
            print(line)
            examples = p_cn.read_chinese_examples(
                line_list=[line], is_training=True, 
                first_answer_only=True, 
                replace_mask="[unused1]",
                do_lower_case=True,
                remove_query_in_passage=True)

            print("length of examples", len(examples))
            print(examples[0])
            print(examples[0].doc_tokens[examples[0].start_positions])
            print(examples[0].doc_tokens[examples[0].end_positions])

            train_features = p_cn.convert_chinese_examples_to_features(
                examples=examples,
                tokenizer=tokenizer,
                max_seq_length=384,
                doc_stride=128,
                max_query_length=64,
                is_training=True,
                first_answer_only=True)

            print(train_features[0])
            break

    
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_seq_length", default=384, type=int,
                            help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                                 "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--doc_stride", default=128, type=int,
                        help="When splitting up a long document into chunks, "
                                "how much stride to take between chunks.")
    parser.add_argument("--max_query_length", default=128, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                                "be truncated to this length.")
    parser.add_argument('--do_lower_case', type=bool, default=True)
    parser.add_argument('--remove_query_in_passage', type=bool, default=True)
    parser.add_argument('--enqueue_thread_num', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--window_size', type=int, default=5)
    parser.add_argument('--lmb', type=float, default=0.5)
    parser.add_argument('--train_file', type=str, 
        default="/Users/noble6emc2/Desktop/Tencent/BLANC/code/src/lm/BLANC/test_train.txt")
    parser.add_argument('--model', type=str, default="bert-base-chinese")
    args = parser.parse_args()
    
    for batch in cn_dataloader.get_training_batch_chinese(args, co_training = True):
        input()