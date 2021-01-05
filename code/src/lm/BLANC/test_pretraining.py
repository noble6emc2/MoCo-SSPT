import run_mrqa_blanc_pretraining_chinese as p_cn
from pytorch_pretrained_bert.tokenization import BasicTokenizer, BertTokenizer
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
                replace_mask="[UNK]",
                do_lower_case=True)

            print("length of examples", len(examples))
            print(examples[0])
            print(examples[0].doc_tokens[examples[0].start_positions[0]])
            print(examples[0].doc_tokens[examples[0].end_positions[0]])

            train_features = p_cn.convert_chinese_examples_to_features(
                examples=examples,
                tokenizer=tokenizer,
                max_seq_length=384,
                doc_stride=128,
                max_query_length=64,
                is_training=True)

            #print(examples)