import run_mrqa_blanc_pretraining_chinese as p_cn
from pytorch_pretrained_bert.tokenization import BasicTokenizer, BertTokenizer
import pretrain_dataloader_chinese as cn_dataloader
import mrqa_official_eval as m_eval
import argparse
import unicodedata
from collections import Counter
if __name__ == "__main__":
    '''

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
    '''
    print(m_eval.f1_score("normalize_answer", "lize_answer"))
    print(m_eval.normalize_answer("oh adasa shoot").split())
    print(Counter("normalize_answer"))
    print(Counter("lize_answer"))
    print(Counter("normalize_answer") & Counter("lize_answer"))
    input()
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

    examples, datasets = p_cn.read_crmc_examples(
                "/Users/noble6emc2/Downloads/cmrc2018_dev.json", is_training=True, 
                first_answer_only=True, 
                do_lower_case=True,
                remove_query_in_passage=False)
    print(len(examples))

    tokenizer = BertTokenizer.from_pretrained(
            "bert-base-chinese")
    print(len(examples))
    for e, d in zip(examples, datasets):
        print(e)
        print(d)
        input("---------")
    train_features = p_cn.convert_chinese_examples_to_features(
                examples=examples,
                tokenizer=tokenizer,
                max_seq_length=384,
                doc_stride=128,
                max_query_length=64,
                is_training=True,
                first_answer_only=True)
    print(len(train_features))
    for i in train_features:
        print(i)
        input("---------")
    #for batch in cn_dataloader.get_training_batch_chinese(args, co_training = True):
    #    input()
    
    '''
    doc_tokens = []
    paragraph_text = "《纯情罗曼史》是一部由漫画家中村春菊所绘的BL漫画，后来被改编为广播剧、小说、电视动画以及游戏。与同为中村春菊所作之漫画「世界第一初恋」享有共同世界。《纯情罗曼史》内容分为三部分：此漫画的总销量目前己超过四百万册，在同类型的漫画作品里可说是异数。此系列的改编小说《纯爱罗曼史》和《自我中心纯爱》声称是「把漫画主角之一创作的妄想小说搬到现实来」的小说。生日:11月22日　血型：A型。 属性:傲娇受。同名电视动画于2007年宣布开始制作，于2008年4月开始播放。第二季电视动画则于2008年10月开始播放，片尾插图由中村春菊担任，动画风景制作则使用实景照片。相隔6年之后，于2014年7月宣布决定制作第3季，预定2015年7月开始播放。"
    raw_doc_tokens = p_cn.tokenize_chinese(paragraph_text, masking = None, do_lower_case=True)
    print(raw_doc_tokens)
    char_to_word_offset = []
    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True

        cat = unicodedata.category(c)
        if cat == "Zs":
            return True
            
        return False
    k = 0
    temp_word = ""
    
    for c in paragraph_text:
        if is_whitespace(c):
            char_to_word_offset.append(k - 1)
            continue
        else:
            temp_word += c
            char_to_word_offset.append(k)

        if True:
            temp_word = temp_word.lower()

        if temp_word == raw_doc_tokens[k]:
            doc_tokens.append(temp_word)
            temp_word = ""
            k += 1

    print(k, len(raw_doc_tokens))
    print(temp_word)
    print(raw_doc_tokens[k])
    '''