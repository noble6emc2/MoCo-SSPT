import logging
import unicodedata
import collections
import json
import re
import gzip
import numpy as np
import six
from pytorch_pretrained_bert.tokenization import BasicTokenizer, BertTokenizer, whitespace_tokenize
from pytorch_pretrained_bert.tokenization import _is_punctuation, _is_whitespace, _is_control

class SquadExample(object):
    """
    A single training/test example for the Squad dataset.
    For examples without an answer, the start and end position are -1.
    """

    def __init__(self,
                 qas_id,
                 question_text,
                 doc_tokens,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None,
                 is_impossible=None,
                 start_positions=None,
                 end_positions=None,
                 orig_answer_texts=None):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible
    
        self.start_positions = start_positions
        self.end_positions = end_positions
        self.orig_answer_texts = orig_answer_texts

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (self.qas_id)
        s += ", question_text: %s" % (
            self.question_text)
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
        if self.start_position:
            s += ", start_position: %d" % (self.start_position)
        if self.end_position:
            s += ", end_position: %d" % (self.end_position)
        if self.is_impossible:
            s += ", is_impossible: %r" % (self.is_impossible)
        return s


class MRQAExample(object):
    """
    A single training/test example for the MRQA dataset.
    For examples without an answer, the start and end position are -1.
    """

    def __init__(self,
                 qas_id,
                 question_text,
                 doc_tokens,
                 #paragraph_text,
                 orig_answer_text=None,
                 start_positions=None,
                 end_positions=None,
                 start_position=None,
                 end_position=None,
                 is_impossible=None):
        self.qas_id = qas_id
        self.question_text = question_text
        #self.paragraph_text = paragraph_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_positions = start_positions
        self.end_positions = end_positions
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (self.qas_id)
        s += ", question_text: %s" % (
            self.question_text)
        s += ", doc_tokens: [%s]" % ("| ".join(self.doc_tokens))
        #s += ", paragraph_text: %s" % (
        #    self.paragraph_text)
        if self.start_positions:
            s += ", start_positions: %s" % (self.start_positions)
        if self.end_positions:
            s += ", end_positions: %s" % (self.end_positions)
        if self.is_impossible:
            s += ", is_impossible: %d" % (self.is_impossible)
        return s


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids,
                 start_positions=None,
                 end_positions=None,
                 start_position=None,
                 end_position=None,
                 is_impossible=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_positions = start_positions
        self.end_positions = end_positions
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "*** Feature ***"
        s += "\nunique_id: %s" % (self.unique_id)
        s += "\nexample_index: %s" % (self.example_index)
        s += "\ndoc_span_index: %s" % (self.doc_span_index)
        s += "\ntokens: %s" % (" ".join(self.tokens))
        s += "\ntoken_to_orig_map: %s" % (" ".join([
            "%d:%d" % (x, y) for (x, y) in self.token_to_orig_map.items()]))
        s += "\ntoken_is_max_context: %s" % (" ".join([
            "%d:%s" % (x, y) for (x, y) in self.token_is_max_context.items()
        ]))
        s += "\ninput_ids: %s" % " ".join([str(x) for x in self.input_ids])
        s += "\ninput_mask: %s" % " ".join([str(x) for x in self.input_mask])
        s += "\nsegment_ids: %s" % " ".join([str(x) for x in self.segment_ids])
        if type(self.start_positions) == int:
            s += "\nanswer_span: "+ " ".join(
                self.tokens[self.start_positions:(self.end_positions + 1)])
        else:
            s += "\nanswer_span: "+ " ".join(
                self.tokens[self.start_positions[0]:(self.end_positions[0] + 1)])

        s += "\nstart_positions: %s" % (self.start_positions)
        s += "\nend_positions: %s" % (self.end_positions)
        return s

class BaseProcessor(object):
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                            orig_answer_text):
        """Returns tokenized answer spans that better match the annotated answer."""
        tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

        for new_start in range(input_start, input_end + 1):
            for new_end in range(input_end, new_start - 1, -1):
                text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
                if text_span == tok_answer_text:
                    return (new_start, new_end)

        return (input_start, input_end)

    @staticmethod
    def _check_is_max_context(doc_spans, cur_span_index, position):
        """Check if this is the 'max context' doc span for the token."""
        best_score = None
        best_span_index = None
        for (span_index, doc_span) in enumerate(doc_spans):
            end = doc_span.start + doc_span.length - 1
            if position < doc_span.start:
                continue
            if position > end:
                continue
            num_left_context = position - doc_span.start
            num_right_context = end - position
            score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
            if best_score is None or score > best_score:
                best_score = score
                best_span_index = span_index

        return cur_span_index == best_span_index

    def _convert_examples_to_features(self, examples, tokenizer, max_seq_length,
                                doc_stride, max_query_length, is_training,
                                first_answer_only):
        """Loads a data file into a list of `InputBatch`s."""

        unique_id = 1000000000

        features = []
        for (example_index, example) in enumerate(examples):
            query_tokens = tokenizer.tokenize(example.question_text)

            if len(query_tokens) > max_query_length:
                query_tokens = query_tokens[0:max_query_length]

            tok_to_orig_index = []
            orig_to_tok_index = []
            all_doc_tokens = []
            for (i, token) in enumerate(example.doc_tokens):
                orig_to_tok_index.append(len(all_doc_tokens))
                sub_tokens = tokenizer.tokenize(token)
                for sub_token in sub_tokens:
                    tok_to_orig_index.append(i)
                    all_doc_tokens.append(sub_token)

            tok_start_positions = []
            tok_end_positions = []
            
            if type(example.start_positions) != list:
                temp_start_positions = [example.start_positions]
                temp_end_positions = [example.end_positions]
            else:
                temp_start_positions = example.start_positions
                temp_end_positions = example.end_positions

            for start_position, end_position in zip(temp_start_positions, temp_end_positions):
                tok_start_position = -1
                tok_end_position = -1
                tok_start_position = orig_to_tok_index[start_position]
                if end_position < len(example.doc_tokens) - 1:
                    tok_end_position = orig_to_tok_index[end_position + 1] - 1
                else:
                    tok_end_position = len(all_doc_tokens) - 1

                (tok_start_position, tok_end_position) = _improve_answer_span(
                    all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
                    example.orig_answer_text)

                tok_start_positions.append(tok_start_position)
                tok_end_positions.append(tok_end_position)

            # The -3 accounts for [CLS], [SEP] and [SEP]
            max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

            # We can have documents that are longer than the maximum sequence length.
            # To deal with this we do a sliding window approach, where we take chunks
            # of the up to our max length with a stride of `doc_stride`.
            _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
                "DocSpan", ["start", "length"])
            doc_spans = []
            start_offset = 0
            while start_offset < len(all_doc_tokens):
                length = len(all_doc_tokens) - start_offset
                if length > max_tokens_for_doc:
                    length = max_tokens_for_doc
                doc_spans.append(_DocSpan(start=start_offset, length=length))
                if start_offset + length == len(all_doc_tokens):
                    break
                start_offset += min(length, doc_stride)
            
            for (doc_span_index, doc_span) in enumerate(doc_spans):
                tokens = []
                token_to_orig_map = {}
                token_is_max_context = {}
                segment_ids = []
                tokens.append("[CLS]")
                segment_ids.append(0)
                for token in query_tokens:
                    tokens.append(token)
                    segment_ids.append(0)
                tokens.append("[SEP]")
                segment_ids.append(0)

                for i in range(doc_span.length):
                    split_token_index = doc_span.start + i
                    token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

                    is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                        split_token_index)
                    token_is_max_context[len(tokens)] = is_max_context
                    tokens.append(all_doc_tokens[split_token_index])
                    segment_ids.append(1)
                tokens.append("[SEP]")
                segment_ids.append(1)

                input_ids = tokenizer.convert_tokens_to_ids(tokens)

                # The mask has 1 for real tokens and 0 for padding tokens. Only real
                # tokens are attended to.
                input_mask = [1] * len(input_ids)

                # Zero-pad up to the sequence length.
                while len(input_ids) < max_seq_length:
                    input_ids.append(0)
                    input_mask.append(0)
                    segment_ids.append(0)

                assert len(input_ids) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_ids) == max_seq_length

                start_position = None
                end_position = None
                
                # For training, if our document chunk does not contain an annotation
                # we throw it out, since there is nothing to predict.
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                start_positions = []
                end_positions = []
                for tok_start_position, token_end_position in zip(tok_start_positions, tok_end_positions):
                    out_of_span = False
                    if not (tok_start_position >= doc_start and
                            tok_end_position <= doc_end):
                        out_of_span = True

                    if out_of_span or example.is_impossible:
                        start_position = 0
                        end_position = 0
                    else:
                        doc_offset = len(query_tokens) + 2
                        start_position = tok_start_position - doc_start + doc_offset
                        end_position = tok_end_position - doc_start + doc_offset
                    
                    start_positions.append(start_position)
                    end_positions.append(end_position)
                
                multimatch_start_labels = np.zeros((max_seq_length,))
                multimatch_end_labels = np.zeros((max_seq_length,))
                multimatch_start_labels[start_positions] = 1
                multimatch_end_labels[end_positions] = 1


                if example_index < 0:
                    self.self.logger.info("*** Example ***")
                    self.self.logger.info("unique_id: %s" % (unique_id))
                    self.self.logger.info("example_index: %s" % (example_index))
                    self.logger.info("doc_span_index: %s" % (doc_span_index))
                    self.logger.info("tokens: %s" % " ".join(tokens))
                    self.logger.info("token_to_orig_map: %s" % " ".join([
                        "%d:%d" % (x, y) for (x, y) in token_to_orig_map.items()]))
                    self.logger.info("token_is_max_context: %s" % " ".join([
                        "%d:%s" % (x, y) for (x, y) in token_is_max_context.items()
                    ]))
                    self.logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                    self.logger.info(
                        "input_mask: %s" % " ".join([str(x) for x in input_mask]))
                    self.logger.info(
                        "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                    if is_training:
                        answer_text = " ".join(tokens[start_positions[0]:(end_positions[0] + 1)])
                        self.logger.info("start_positions: %s" % (start_positions))
                        self.logger.info("end_positions: %s" % (end_positions))
                        self.logger.info("start_labels: %s" % (multimatch_start_labels))
                        self.logger.info("end_labels: %s" % (multimatch_end_labels))
                        self.logger.info(
                            "answer: %s" % (answer_text))

                if first_answer_only:
                    start_positions = start_positions[0]
                    end_positions = end_positions[0]

                features.append(
                    InputFeatures(
                        unique_id=unique_id,
                        example_index=example_index,
                        doc_span_index=doc_span_index,
                        tokens=tokens, # Query + Passage Span
                        token_to_orig_map=token_to_orig_map,
                        token_is_max_context=token_is_max_context,
                        input_ids=input_ids, # Mask for max seq length
                        input_mask=input_mask, # Vocab id list
                        segment_ids=segment_ids,
                        start_positions=start_positions, # Answer start pos(Query included)
                        end_positions=end_positions,
                        start_position=start_positions,
                        end_position=end_positions,))
                unique_id += 1

        return features

    def tokenize_chinese(self, text, masking, do_lower_case):
        temp_x = ""
        text = convert_to_unicode(text)
        idx = 0
        if do_lower_case:
            text = text.lower()

        while(idx < len(text)):
            c = text[idx]
            if masking is not None and text[idx: idx + len(masking)] == masking.lower():
                temp_x += masking
                idx += len(masking)
                continue
            elif BasicTokenizer._is_chinese_char(ord(c)) or _is_punctuation(c) or \
                    _is_whitespace(c) or _is_control(c):
                temp_x += " " + c + " "
            else:
                temp_x += c
            
            idx += 1

        return temp_x.split()

    def convert_to_unicode(self, text):
        """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
        if six.PY3:
            if isinstance(text, str):
                return text
            elif isinstance(text, bytes):
                return text.decode("utf-8", "ignore")
            else:
                raise ValueError("Unsupported string type: %s" % (type(text)))
        elif six.PY2:
            if isinstance(text, str):
                return text.decode("utf-8", "ignore")
            elif isinstance(text, unicode):
                return text
            else:
                raise ValueError("Unsupported string type: %s" % (type(text)))
        else:
            raise ValueError("Not running on Python2 or Python 3?")
    
    def convert_english_examples_to_features(self, *args, **kwargs):
        return self._convert_examples_to_features(*args, **kwargs)

    def convert_chinese_examples_to_features(self, *args, **kwargs):
        return self._convert_examples_to_features(*args, **kwargs)

class PretrainingProcessor(BaseProcessor):
    def read_chinese_examples(self, line_list, is_training, 
        first_answer_only, replace_mask, do_lower_case,
        remove_query_in_passage):
        """Read a Chinese json file for pretraining into a list of MRQAExample."""
        input_data = [json.loads(line) for line in line_list] #.decode('utf-8').strip()

        def is_whitespace(c):
            if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
                return True

            cat = unicodedata.category(c)
            if cat == "Zs":
                return True
                
            return False

        examples = []
        num_answers = 0
        for i, article in enumerate(input_data):
            for entry in article["paragraphs"]:
                paragraph_text = entry["context"].strip()
                raw_doc_tokens = self.tokenize_chinese(paragraph_text, 
                        masking = None,
                        do_lower_case= do_lower_case)
                doc_tokens = []
                char_to_word_offset = []
                prev_is_whitespace = True
                
                k = 0
                temp_word = ""
                for c in paragraph_text:
                    if is_whitespace(c):
                        char_to_word_offset.append(k - 1)
                        continue
                    else:
                        temp_word += c
                        char_to_word_offset.append(k)

                    if do_lower_case:
                        temp_word = temp_word.lower()

                    if temp_word == raw_doc_tokens[k]:
                        doc_tokens.append(temp_word)
                        temp_word = ""
                        k += 1

                if k != len(raw_doc_tokens):
                    self.logger.info("Warning: paragraph '{}' tokenization error ".format(paragraph_text))
                    continue

                for qa in entry["qas"]:
                    qas_id = article["id"] + "_" + entry["id"] + "_" + qa["id"]
                    if qa["question"].find('UNK') == -1:
                        print(f"WARNING: Cannot Find UNK in Question %s" % qas_id)
                        continue

                    if remove_query_in_passage:
                        query = qa["question"]
                        mask_start = query.find("UNK")
                        mask_end = mask_start + 3
                        pattern = (re.escape(unicodedata.normalize('NFKC', query[:mask_start].strip())) + 
                            ".*" + re.escape(unicodedata.normalize('NFKC', query[mask_end:].strip())))
                        if re.search(pattern,
                            unicodedata.normalize('NFKC', paragraph_text)) is not None:
                            #print(f"WARNING: Query in Passage Detected in Question %s" % qas_id)
                            #print("Question", query, "Passage", paragraph_text)
                            continue

                    question_text = qa["question"].replace("UNK", replace_mask)
                    is_impossible = qa.get('is_impossible', False)
                    start_position = None
                    end_position = None
                    orig_answer_text = None
                    
                    answers = qa["answers"]
                    num_answers += sum([len(spans['answer_all_start']) for spans in answers])
                    
                    # import ipdb
                    # ipdb.set_trace()
                    spans = sorted([[start, start + len(ans['text']) - 1, ans['text']] 
                        for ans in answers for start in ans['answer_all_start']])
                    #print("spans", spans)
                    # take first span
                    if first_answer_only:
                        include_span_num = 1
                    else:
                        include_span_num = len(spans)

                    start_positions = []
                    end_positions = []
                    for i in range(min(include_span_num, len(spans))):
                        char_start, char_end, answer_text = spans[i][0], spans[i][1], spans[i][2]
                        orig_answer_text = paragraph_text[char_start:char_end+1]
                        #print("orig_answer_text", orig_answer_text)
                        if orig_answer_text != answer_text:
                            self.logger.info("Answer error: {}, Original {}".format(
                                answer_text, orig_answer_text))
                            continue
                        
                        start_position, end_position = char_to_word_offset[char_start], char_to_word_offset[char_end]
                        #print("start_position", "end_position", start_position, end_position)
                        #print("doc_tokens", doc_tokens)
                        start_positions.append(start_position)
                        end_positions.append(end_position)

                    if len(spans) == 0:
                        start_positions.append(0)
                        end_positions.append(0)

                    if first_answer_only:
                        start_positions = start_positions[0]
                        end_positions = end_positions[0]

                    example = MRQAExample(
                        qas_id=qas_id,
                        question_text=question_text, #question
                        #paragraph_text=paragraph_text, # context text
                        doc_tokens=doc_tokens, #passage text
                        orig_answer_text=orig_answer_text, # answer text
                        start_positions=start_positions, #answer start
                        end_positions=end_positions, #answer end
                        is_impossible=is_impossible)
                    examples.append(example)


        #logger.info('Num avg answers: {}'.format(num_answers / len(examples)))
        return examples
        

    def read_english_examples(self, line_list, is_training, 
                first_answer_only, replace_mask, do_lower_case):
        """Read a English json file for pretraining into a list of MRQAExample."""
        input_data = [json.loads(line) for line in line_list] #.decode('utf-8').strip()

        def is_whitespace(c):
            if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
                return True

            cat = unicodedata.category(c)
            if cat == "Zs":
                return True

            return False

        examples = []
        num_answers = 0
        for i, article in enumerate(input_data):
            paragraph_text = article["passage"].strip()
            doc_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True
            for c in paragraph_text:
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                    else:
                        doc_tokens[-1] += c
                    prev_is_whitespace = False
                char_to_word_offset.append(len(doc_tokens) - 1)

            #for a_id, answer in enumerate(article["answers"]):
            qas_id = article["qid"]
            if article["question"].find('[BLANK]') == -1:
                print(f"WARNING: Cannot Find [BLANK] in Question %s" % qas_id)
                continue

            '''if remove_query_in_passage:
                query = qa["question"]
                mask_start = query.find("UNK")
                mask_end = mask_start + 3
                pattern = (re.escape(unicodedata.normalize('NFKC', query[:mask_start].strip())) + 
                    ".*" + re.escape(unicodedata.normalize('NFKC', query[mask_end:].strip())))
                if re.search(pattern,
                    unicodedata.normalize('NFKC', paragraph_text)) is not None:
                    #print(f"WARNING: Query in Passage Detected in Question %s" % qas_id)
                    #print("Question", query, "Passage", paragraph_text)
                    continue'''

            question_text = article["question"].replace("[BLANK]", replace_mask)
            start_position = None
            end_position = None
            orig_answer_text = None
            is_impossible = False
            
            answers = [
                {
                    'text': ans,
                    'answer_all_start': [m.start() 
                        for m in re.finditer(re.escape(ans), paragraph_text)]
                }
                for ans in article["answers"]
                ]
            num_answers += sum([len(spans['answer_all_start']) for spans in answers])
            
            # import ipdb
            # ipdb.set_trace()
            spans = sorted([[start, start + len(ans['text']) - 1, ans['text']] 
                for ans in answers for start in ans['answer_all_start']])
            #print("spans", spans)
            # take first span
            if first_answer_only:
                include_span_num = 1
            else:
                include_span_num = len(spans)

            start_positions = []
            end_positions = []
            #print("spans length:", len(spans))
            #print("article[\"answers\"]", article["answers"])
            #input()
            for i in range(min(include_span_num, len(spans))):
                char_start, char_end, answer_text = spans[i][0], spans[i][1], spans[i][2]
                orig_answer_text = paragraph_text[char_start:char_end+1]
                #print("orig_answer_text", orig_answer_text)
                if orig_answer_text != answer_text:
                    self.logger.info("Answer error: {}, Original {}".format(
                        answer_text, orig_answer_text))
                    continue
                
                start_position, end_position = char_to_word_offset[char_start], char_to_word_offset[char_end]
                #print("start_position", "end_position", start_position, end_position)
                #print("doc_tokens", doc_tokens)
                start_positions.append(start_position)
                end_positions.append(end_position)

            if len(spans) == 0:
                start_positions.append(0)
                end_positions.append(0)
                orig_answer_text = ''
                is_impossible = True

            if first_answer_only:
                start_positions = start_positions[0]
                end_positions = end_positions[0]

            example = MRQAExample(
                qas_id=qas_id,
                question_text=question_text, #question
                #paragraph_text=paragraph_text, # context text
                doc_tokens=doc_tokens, #passage text
                orig_answer_text=orig_answer_text, # answer text
                start_positions=start_positions, #answer start
                end_positions=end_positions, #answer end
                is_impossible=is_impossible
                )
            examples.append(example)


        #self.logger.info('Num avg answers: {}'.format(num_answers / len(examples)))
        return examples


class MRQAProcessor(BaseProcessor):
    def read_mrqa_examples(self, input_file, is_training, 
            first_answer_only, do_lower_case,
            remove_query_in_passage):
        """Read crmc json file for pretraining into a list of MRQAExample."""
        with gzip.GzipFile(input_file, 'r') as reader:
            # skip header
            content = reader.read().decode('utf-8').strip().split('\n')[1:]
            input_data = [json.loads(line) for line in content]

        def is_whitespace(c):
            if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
                return True 
            return False

        examples = []
        num_answers = 0
        datasets = []
        for i, entry in enumerate(input_data):
            if i % 1000 == 0:
                self.logger.info("Processing %d / %d.." % (i, len(input_data)))

            paragraph_text = entry["context"]
            doc_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True
            for c in paragraph_text:
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                    else:
                        doc_tokens[-1] += c
                    prev_is_whitespace = False
                char_to_word_offset.append(len(doc_tokens) - 1)

            for qa in entry["qas"]:
                qas_id = qa["qid"]
                question_text = qa["question"]# .replace("UNK", replace_mask)
                is_impossible = qa.get('is_impossible', False)
                start_position = None
                end_position = None
                orig_answer_text = None
                
                answers = qa["detected_answers"]
                # import ipdb
                # ipdb.set_trace()
                spans = sorted([span for spans in answers for span in spans['char_spans']])
                # take first span
                char_start, char_end = spans[0][0], spans[0][1]
                orig_answer_text = paragraph_text[char_start:char_end+1]
                start_position, end_position = char_to_word_offset[char_start], char_to_word_offset[char_end]
                num_answers += sum([len(spans['char_spans']) for spans in answers])

                example = MRQAExample(
                    qas_id=qas_id,
                    question_text=question_text, #question
                    #paragraph_text=paragraph_text, # context text
                    doc_tokens=doc_tokens, #passage text
                    orig_answer_text=orig_answer_text, # answer text
                    start_positions=start_position, #answer start
                    end_positions=end_position, #answer end
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=is_impossible)
                examples.append(example)

        self.logger.info('Num avg answers: {}'.format(num_answers / len(examples)))
        return examples

    def read_examples(self, *args, **kwargs):
        return self.read_mrqa_examples(*args, **kwargs)

    def convert_mrqa_examples_to_features(self, *args, **kwargs):
        return self._convert_examples_to_features(*args, **kwargs)


class SQuADProcessor(BaseProcessor):
    def read_squad_examples(self, input_file, is_training, version_2_with_negative):
        """Read a SQuAD json file into a list of SquadExample."""
        with open(input_file, "r", encoding='utf-8') as reader:
            input_data = json.load(reader)["data"]

        def is_whitespace(c):
            if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
                return True
            return False

        examples = []
        for entry in input_data:
            for paragraph in entry["paragraphs"]:
                paragraph_text = paragraph["context"]
                doc_tokens = []
                char_to_word_offset = []
                prev_is_whitespace = True
                for c in paragraph_text:
                    if is_whitespace(c):
                        prev_is_whitespace = True
                    else:
                        if prev_is_whitespace:
                            doc_tokens.append(c)
                        else:
                            doc_tokens[-1] += c
                        prev_is_whitespace = False
                    char_to_word_offset.append(len(doc_tokens) - 1)

                for qa in paragraph["qas"]:
                    qas_id = qa["id"]
                    question_text = qa["question"]
                    start_position = None
                    end_position = None
                    orig_answer_text = None
                    is_impossible = False
                    start_positions = []
                    end_positions = []
                    orig_answer_texts = []
                    if is_training:
                        if version_2_with_negative:
                            is_impossible = qa["is_impossible"]
                        if (len(qa["answers"]) != 1) and (not is_impossible):
                            raise ValueError(
                                "For training, each question should have exactly 1 answer.")
                        if not is_impossible:
                            answer = qa["answers"][0]
                            orig_answer_text = answer["text"]
                            answer_offset = answer["answer_start"]
                            answer_length = len(orig_answer_text)
                            start_position = char_to_word_offset[answer_offset]
                            end_position = char_to_word_offset[answer_offset + answer_length - 1]
                            actual_text = " ".join(doc_tokens[start_position:(end_position + 1)])
                            cleaned_answer_text = " ".join(
                                whitespace_tokenize(orig_answer_text))
                            if actual_text.find(cleaned_answer_text) == -1:
                                self.logger.warning("Could not find answer: '%s' vs. '%s'",
                                            actual_text, cleaned_answer_text)
                                continue
                        else:
                            start_position = -1
                            end_position = -1
                            orig_answer_text = ""
                    else:
                        if version_2_with_negative:
                            is_impossible = qa["is_impossible"]
                        if not is_impossible:
                            answers = qa["answers"]
                            for answer in answers:
                                orig_answer_text = answer["text"]
                                answer_offset = answer["answer_start"]
                                answer_length = len(orig_answer_text)
                                start_position = char_to_word_offset[answer_offset]
                                end_position = char_to_word_offset[answer_offset + answer_length - 1]
                                actual_text = " ".join(doc_tokens[start_position:(end_position + 1)])
                                cleaned_answer_text = " ".join(
                                    whitespace_tokenize(orig_answer_text))
                                if actual_text.find(cleaned_answer_text) == -1:
                                    self.logger.warning("Could not find answer: '%s' vs. '%s'",
                                                actual_text, cleaned_answer_text)
                                    continue
                                start_positions.append(start_position)
                                end_positions.append(end_position)
                                orig_answer_texts.append(orig_answer_text)

                        else:
                            start_position = -1
                            end_position = -1
                            orig_answer_text = ""
                            start_positions.append(start_position)
                            end_positions.append(end_position)
                            orig_answer_texts.append(orig_answer_text)

                    example = SquadExample(
                        qas_id=qas_id,
                        question_text=question_text,
                        doc_tokens=doc_tokens,
                        orig_answer_text=orig_answer_text,
                        start_position=start_position,
                        end_position=end_position,
                        is_impossible=is_impossible,
                        start_positions=start_positions,
                        end_positions=end_positions,
                        orig_answer_texts=orig_answer_texts)
                    examples.append(example)
        return examples

    def convert_squad_examples_to_features(self, examples, tokenizer, max_seq_length,
                                 doc_stride, max_query_length, is_training):
        """Loads a data file into a list of `InputBatch`s."""

        unique_id = 1000000000

        features = []
        for (example_index, example) in enumerate(examples):
            query_tokens = tokenizer.tokenize(example.question_text)

            if len(query_tokens) > max_query_length:
                query_tokens = query_tokens[0:max_query_length]

            tok_to_orig_index = []
            orig_to_tok_index = []
            all_doc_tokens = []
            for (i, token) in enumerate(example.doc_tokens):
                orig_to_tok_index.append(len(all_doc_tokens))
                sub_tokens = tokenizer.tokenize(token)
                for sub_token in sub_tokens:
                    tok_to_orig_index.append(i)
                    all_doc_tokens.append(sub_token)

            tok_start_position = None
            tok_end_position = None
            if example.is_impossible:
                tok_start_position = -1
                tok_end_position = -1
            
            if not example.is_impossible:
                tok_start_position = orig_to_tok_index[example.start_position]
                if example.end_position < len(example.doc_tokens) - 1:
                    tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
                else:
                    tok_end_position = len(all_doc_tokens) - 1
                (tok_start_position, tok_end_position) = self._improve_answer_span(
                    all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
                    example.orig_answer_text)

            # The -3 accounts for [CLS], [SEP] and [SEP]
            max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

            _DocSpan = collections.namedtuple(
                "DocSpan", ["start", "length"])
            doc_spans = []
            start_offset = 0
            while start_offset < len(all_doc_tokens):
                length = len(all_doc_tokens) - start_offset
                if length > max_tokens_for_doc:
                    length = max_tokens_for_doc
                doc_spans.append(_DocSpan(start=start_offset, length=length))
                if start_offset + length == len(all_doc_tokens):
                    break
                start_offset += min(length, doc_stride)
            
            for (doc_span_index, doc_span) in enumerate(doc_spans):
                tokens = []
                token_to_orig_map = {}
                token_is_max_context = {}
                segment_ids = []
                tokens.append("[CLS]")
                segment_ids.append(0)
                for token in query_tokens:
                    tokens.append(token)
                    segment_ids.append(0)
                tokens.append("[SEP]")
                segment_ids.append(0)

                for i in range(doc_span.length):
                    split_token_index = doc_span.start + i
                    token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

                    is_max_context = self._check_is_max_context(doc_spans, doc_span_index,
                                                        split_token_index)
                    token_is_max_context[len(tokens)] = is_max_context
                    tokens.append(all_doc_tokens[split_token_index])
                    segment_ids.append(1)
                tokens.append("[SEP]")
                segment_ids.append(1)

                input_ids = tokenizer.convert_tokens_to_ids(tokens)
                input_mask = [1] * len(input_ids)

                while len(input_ids) < max_seq_length:
                    input_ids.append(0)
                    input_mask.append(0)
                    segment_ids.append(0)

                assert len(input_ids) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_ids) == max_seq_length

                start_position = None
                end_position = None
                if not example.is_impossible:
                    doc_start = doc_span.start
                    doc_end = doc_span.start + doc_span.length - 1
                    out_of_span = False
                    if not (tok_start_position >= doc_start and
                            tok_end_position <= doc_end):
                        out_of_span = True
                    if out_of_span:
                        start_position = 0
                        end_position = 0
                    else:
                        doc_offset = len(query_tokens) + 2
                        start_position = tok_start_position - doc_start + doc_offset
                        end_position = tok_end_position - doc_start + doc_offset
                if example.is_impossible:
                    start_position = 0
                    end_position = 0
                    continue
                if example_index < 0:
                    self.logger.info("*** Example ***")
                    self.logger.info("unique_id: %s" % (unique_id))
                    self.logger.info("example_index: %s" % (example_index))
                    self.logger.info("doc_span_index: %s" % (doc_span_index))
                    self.logger.info("tokens: %s" % " ".join(tokens))
                    self.logger.info("token_to_orig_map: %s" % " ".join([
                        "%d:%d" % (x, y) for (x, y) in token_to_orig_map.items()]))
                    self.logger.info("token_is_max_context: %s" % " ".join([
                        "%d:%s" % (x, y) for (x, y) in token_is_max_context.items()
                    ]))
                    self.logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                    self.logger.info(
                        "input_mask: %s" % " ".join([str(x) for x in input_mask]))
                    self.logger.info(
                        "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                    if is_training and example.is_impossible:
                        self.logger.info("impossible example")
                    if is_training and not example.is_impossible:
                        answer_text = " ".join(tokens[start_position:(end_position + 1)])
                        self.logger.info("start_position: %d" % (start_position))
                        self.logger.info("end_position: %d" % (end_position))
                        self.logger.info(
                            "answer: %s" % (answer_text))

                features.append(
                    InputFeatures(
                        unique_id=unique_id,
                        example_index=example_index,
                        doc_span_index=doc_span_index,
                        tokens=tokens,
                        token_to_orig_map=token_to_orig_map,
                        token_is_max_context=token_is_max_context,
                        input_ids=input_ids,
                        input_mask=input_mask,
                        segment_ids=segment_ids,
                        start_position=start_position,
                        end_position=end_position,
                        is_impossible=example.is_impossible))
                unique_id += 1

        return features

    def read_examples(self, *args, **kwargs):
        return self.read_squad_examples(*args, **kwargs)

class CMRCProcessor(BaseProcessor):
    def read_cmrc_examples(self, input_file, is_training, 
            first_answer_only, do_lower_case,
            remove_query_in_passage):
        """Read crmc json file for pretraining into a list of MRQAExample."""
        with open(input_file, 'r', encoding='utf-8') as fin:
            # skip header
            input_js = json.load(fin)
            input_data = input_js["data"]

        def is_whitespace(c):
            if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
                return True

            cat = unicodedata.category(c)
            if cat == "Zs":
                return True
                
            return False

        examples = []
        num_answers = 0
        datasets = []
        for i, article in enumerate(input_data):
            for entry in article["paragraphs"]:
                paragraph_text = entry["context"].strip()
                raw_doc_tokens = self.tokenize_chinese(paragraph_text, 
                        masking = None,
                        do_lower_case= do_lower_case)
                doc_tokens = []
                char_to_word_offset = []
                prev_is_whitespace = True
                
                k = 0
                temp_word = ""
                for c in paragraph_text:
                    if is_whitespace(c):
                        char_to_word_offset.append(k - 1)
                        continue
                    else:
                        temp_word += c
                        char_to_word_offset.append(k)

                    if do_lower_case:
                        temp_word = temp_word.lower()

                    if temp_word == raw_doc_tokens[k]:
                        doc_tokens.append(temp_word)
                        temp_word = ""
                        k += 1

                if k != len(raw_doc_tokens):
                    self.logger.info("Warning: paragraph '{}' tokenization error'{}'".format(paragraph_text))
                    continue

                for qa in entry["qas"]:
                    qas_id = qa["id"]
                    '''if qa["question"].find('UNK') == -1:
                        logger.info(f"WARNING: Cannot Find UNK in Question %s" % qas_id)
                        continue'''

                    if remove_query_in_passage:
                        query = qa["question"]
                        mask_start = query.find("UNK")
                        mask_end = mask_start + 3
                        pattern = (re.escape(unicodedata.normalize('NFKC', query[:mask_start].strip())) + 
                            ".*" + re.escape(unicodedata.normalize('NFKC', query[mask_end:].strip())))
                        if re.search(pattern,
                            unicodedata.normalize('NFKC', paragraph_text)) is not None:
                            #print(f"WARNING: Query in Passage Detected in Question %s" % qas_id)
                            #print("Question", query, "Passage", paragraph_text)
                            continue

                    question_text = qa["question"]# .replace("UNK", replace_mask)
                    is_impossible = qa.get('is_impossible', False)
                    start_position = None
                    end_position = None
                    orig_answer_text = None
                    
                    answers = qa["answers"]
                    num_answers += len(answers)
                    
                    # import ipdb
                    # ipdb.set_trace()
                    '''
                    [
                        [ans['answer_start'] + 1, 
                        ans['answer_start'] + len(ans['text']), 
                        ans['text']] 
                        for ans in answers]
                    '''
                    ans_list = []
                    answer_text_list = []
                    for ans in answers:
                        answer_text_list.append(ans['text'])
                        if ans['answer_start'] == -1:
                            continue

                        if paragraph_text[ans['answer_start']:].startswith(ans['text']):
                            ans_list.append([ans['answer_start'], 
                                ans['answer_start'] + len(ans['text']) - 1, 
                                ans['text']]
                                )
                        elif paragraph_text[ans['answer_start'] - 2:].startswith(ans['text']):
                            ans_list.append([ans['answer_start'] - 2, 
                                ans['answer_start'] + len(ans['text']) - 3, 
                                ans['text']]
                                )
                        else:
                            ans_list.append([ans['answer_start'] - 1, 
                                ans['answer_start'] + len(ans['text']) - 2, 
                                ans['text']]
                                )

                    spans = sorted(ans_list)
                    #print("spans", spans)
                    # take first span
                    if first_answer_only:
                        include_span_num = 1
                    else:
                        include_span_num = len(spans)

                    start_positions = []
                    end_positions = []
                    for i in range(min(include_span_num, len(spans))):
                        char_start, char_end, answer_text = spans[i][0], spans[i][1], spans[i][2]
                        orig_answer_text = paragraph_text[char_start:char_end+1]
                        #print("orig_answer_text", orig_answer_text)
                        if orig_answer_text != answer_text:
                            self.logger.info("Answer error: {}, Original {}".format(
                                answer_text, orig_answer_text))
                            print(paragraph_text[char_start:])
                            continue
                        
                        start_position, end_position = char_to_word_offset[char_start], char_to_word_offset[char_end]
                        #print("start_position", "end_position", start_position, end_position)
                        #print("doc_tokens", doc_tokens)
                        start_positions.append(start_position)
                        end_positions.append(end_position)

                    if len(spans) == 0:
                        start_positions.append(0)
                        end_positions.append(0)

                    if first_answer_only:
                        start_positions = start_positions[0]
                        end_positions = end_positions[0]

                    datasets.append({'qid': qas_id, 'answers': answer_text_list})

                    example = MRQAExample(
                        qas_id=qas_id,
                        question_text=question_text, #question
                        #paragraph_text=paragraph_text, # context text
                        doc_tokens=doc_tokens, #passage text
                        orig_answer_text=orig_answer_text, # answer text
                        start_positions=start_positions, #answer start
                        end_positions=end_positions, #answer end
                        start_position=start_positions,
                        end_position=end_positions,
                        is_impossible=is_impossible)
                    examples.append(example)


        #logger.info('Num avg answers: {}'.format(num_answers / len(examples)))
        return examples, datasets

    def read_examples(self, *args, **kwargs):
        return self.read_cmrc_examples(*args, **kwargs)