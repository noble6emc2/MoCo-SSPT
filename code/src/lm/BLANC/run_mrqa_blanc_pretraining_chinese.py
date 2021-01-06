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
from torch import nn
from io import open

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.blanc import BLANC
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
from pytorch_pretrained_bert.tokenization import BasicTokenizer, BertTokenizer
from pytorch_pretrained_bert.tokenization import _is_punctuation, _is_whitespace, _is_control
from mrqa_official_eval import exact_match_score, f1_score, metric_max_over_ground_truths

PRED_FILE = "predictions.json"
EVAL_FILE = "eval_results.txt"
TEST_FILE = "test_results.txt"

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_to_unicode(text):
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

def tokenize_chinese(text, masking, do_lower_case):
    temp_x = ""
    text = convert_to_unicode(text)
    idx = 0
    if do_lower_case:
        text = text.lower()

    while(idx < len(text)):
        c = text[idx]
        if text[idx: idx + len(masking)] == masking.lower():
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
                 is_impossible=None):
        self.qas_id = qas_id
        self.question_text = question_text
        #self.paragraph_text = paragraph_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_positions = start_positions
        self.end_positions = end_positions
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
                 end_positions=None):
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


def read_chinese_examples(line_list, is_training, 
    first_answer_only, replace_mask, do_lower_case):
    """Read a Chinese json file for pretraining into a list of MRQAExample."""
    input_data = [json.loads(line) for line in line_list] #.decode('utf-8').strip()

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    examples = []
    num_answers = 0
    for i, article in enumerate(input_data):
        for entry in article["paragraphs"]:
            paragraph_text = entry["context"].strip()
            raw_doc_tokens = tokenize_chinese(paragraph_text, 
                    masking = "[UNK]",
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
                logger.info("Warning: paragraph '{}' tokenization error ".format(paragraph_text))
                continue

            for qa in entry["qas"]:
                qas_id = article["id"] + "_" + entry["id"] + "_" + qa["id"]
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
                        logger.info("Answer error: {}, Original {}".format(
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


def convert_chinese_examples_to_features(examples, tokenizer, max_seq_length,
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
                logger.info("*** Example ***")
                logger.info("unique_id: %s" % (unique_id))
                logger.info("example_index: %s" % (example_index))
                logger.info("doc_span_index: %s" % (doc_span_index))
                logger.info("tokens: %s" % " ".join(tokens))
                logger.info("token_to_orig_map: %s" % " ".join([
                    "%d:%d" % (x, y) for (x, y) in token_to_orig_map.items()]))
                logger.info("token_is_max_context: %s" % " ".join([
                    "%d:%s" % (x, y) for (x, y) in token_is_max_context.items()
                ]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info(
                    "input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                if is_training:
                    answer_text = " ".join(tokens[start_positions[0]:(end_positions[0] + 1)])
                    logger.info("start_positions: %s" % (start_positions))
                    logger.info("end_positions: %s" % (end_positions))
                    logger.info("start_labels: %s" % (multimatch_start_labels))
                    logger.info("end_labels: %s" % (multimatch_end_labels))
                    logger.info(
                        "answer: %s" % (answer_text))

            if first_answer_only:
                start_positions = start_positions[0]
                end_positions = end_positions[0]

            features.append(
                InputFeatures(
                    unique_id=str(example.qas_id)+"_"+str(doc_span_index),
                    example_index=example.qas_id,
                    doc_span_index=doc_span_index,
                    tokens=tokens, # Query + Passage Span
                    token_to_orig_map=token_to_orig_map,
                    token_is_max_context=token_is_max_context,
                    input_ids=input_ids, # Mask for max seq length
                    input_mask=input_mask, # Vocab id list
                    segment_ids=segment_ids,
                    start_positions=start_positions, # Answer start pos(Query included)
                    end_positions=end_positions,))
            unique_id += 1

    return features


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


RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits"])


def make_predictions(all_examples, all_features, all_results, n_best_size,
                     max_answer_length, do_lower_case, verbose_logging):
    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)
    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result
    _PrelimPrediction = collections.namedtuple(
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]
        prelim_predictions = []
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            for start_index in start_indexes:
                for end_index in end_indexes:
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index]))
        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True)

        _NbestPrediction = collections.namedtuple(
            "NbestPrediction", ["text", "start_logit", "end_logit", "start_index", "end_index"])
        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
            if pred.start_index > 0:
                tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
                tok_text = " ".join(tok_tokens)
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)
                final_text = get_final_text(tok_text, orig_text, do_lower_case, verbose_logging)
                if final_text in seen_predictions:
                    continue
                seen_predictions[final_text] = True
            else:
                final_text = ""
                seen_predictions[final_text] = True

            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit,
                    start_index=orig_doc_start,
                    end_index=orig_doc_end))

        if not nbest:
            nbest.append(
                _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0, start_index=None, end_index=None))
        assert len(nbest) >= 1

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                if entry.text:
                    best_non_null_entry = entry
    
        target_entry = {}
        target_entry["text"] = best_non_null_entry.text
        target_entry["start_logit"] = best_non_null_entry.start_logit
        target_entry["end_logit"] = best_non_null_entry.end_logit
        target_entry["start_index"] = best_non_null_entry.start_index
        target_entry["end_index"] = best_non_null_entry.end_index
        
        probs = _compute_softmax(total_scores)
        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            nbest_json.append(output)

        assert len(nbest_json) >= 1
        all_predictions[example.qas_id] = target_entry
        all_nbest_json[example.qas_id] = nbest_json

    return all_predictions, all_nbest_json


def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):
    """Project the tokenized prediction back to the original text."""

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
    tok_text = " ".join(tokenizer.tokenize(orig_text))
    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if verbose_logging:
            logger.info(
                "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if verbose_logging:
            logger.info("Length not equal after stripping spaces: '%s' vs '%s'",
                        orig_ns_text, tok_ns_text)
        return orig_text

    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if verbose_logging:
            logger.info("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if verbose_logging:
            logger.info("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs


def get_raw_scores(dataset, predictions, examples):
    answers = {}
    for example in dataset:
        for qa in example['qas']:
            answers[qa['qid']] = qa['answers']
    exact_scores = {}
    f1_scores = {}
    scores = {}
    precision_scores = {}
    recall_scores = {}
    for qid, ground_truths in answers.items():
        if qid not in predictions:
            print('Missing prediction for %s' % qid)
            continue
        prediction = predictions[qid]['text']
        exact_scores[qid] = metric_max_over_ground_truths(
            exact_match_score, prediction, ground_truths)[0]
        scores[qid] = metric_max_over_ground_truths(
            f1_score, prediction, ground_truths)
        f1_scores[qid] = scores[qid][0]
        precision_scores[qid] = scores[qid][1]
        recall_scores[qid] = scores[qid][2]
    
    def get_precision(sp, ep, sr, er):
        p_span = set(list(range(sp, ep + 1)))
        r_span = set(list(range(sr, er + 1)))
        return 1.0 * len(p_span & r_span) / len(p_span)
    
    def get_recall(sp, ep, sr, er):
        p_span = set(list(range(sp, ep + 1)))
        r_span = set(list(range(sr, er + 1)))
        return 1.0 * len(p_span & r_span) / len(r_span)
        
    def get_f1(sp, ep, sr, er):
        p = get_precision(sp, ep, sr, er)
        r = get_recall(sp, ep, sr, er)
        if p < 1e-10 or r < 1e-10:
            return 0.0
        else:
            return 2.0 * p * r / (p + r)
    span_f1 = {}
    span_exact = {}
    span_precision = {}
    span_recall = {}
    for example in examples:
        qid = example.qas_id
        if qid not in predictions:
            continue
        sg = example.start_position
        eg = example.end_position
        
        sf = predictions[qid]["start_index"]
        ef = predictions[qid]["end_index"]
        if sf == None:
            sf = -1
        if ef == None:
            ef = -1
        span_f1[qid] = get_f1(sf, ef, sg, eg)
        if sf == sg and ef == eg:
            span_exact[qid] = 1.0
        else:
            span_exact[qid] = 0.0
        span_precision[qid] = get_precision(sf, ef, sg, eg)
        span_recall[qid] = get_recall(sf, ef, sg, eg)

    return exact_scores, \
            f1_scores, \
            precision_scores, \
            recall_scores, \
            span_exact, \
            span_f1, \
            span_precision, \
            span_recall


def make_eval_dict(exact_scores, f1_scores, precision={}, recall = {}, span_exact={}, span_f1={}, span_p={}, span_r={}, qid_list=None):
    if not qid_list:
        total = len(exact_scores)
        return collections.OrderedDict([
            ('exact', 100.0 * sum(exact_scores.values()) / total),
            ('f1', 100.0 * sum(f1_scores.values()) / total),
            ('precision', 100.0 * sum(precision.values()) / total),
            ('recall', 100.0 * sum(recall.values()) / total),
            ('span_exact', 100.0 * sum(span_exact.values()) / total),
            ('span_f1', 100.0 * sum(span_f1.values()) / total),
            ('span_precision', 100.0 * sum(span_p.values()) / total),
            ('span_recall', 100.0 * sum(span_r.values()) / total),
            ('total', total),
        ])
    else:
        total = len(qid_list)
        return collections.OrderedDict([
            ('exact', 100.0 * sum(exact_scores[k] for k in qid_list) / total),
            ('f1', 100.0 * sum(f1_scores[k] for k in qid_list) / total),
            ('precision', 100.0 * sum(precision_scores[k] for k in qid_list) / total),
            ('recall', 100.0 * sum(recall_scores[k] for k in qid_list) / total),
            ('span_exact', 100.0 * sum(span_exact.values()) / total),
            ('span_f1', 100.0 * sum(span_f1.values()) / total),
            ('span_precision', 100.0 * sum(span_p.values()) / total),
            ('span_recall', 100.0 * sum(span_r.values()) / total),
            ('total', total),
        ])


def evaluate(args, model, device, eval_dataset, eval_dataloader,
             eval_examples, eval_features, verbose=True):
    all_results = []
    model.eval()
    for input_ids, input_mask, segment_ids, example_indices in eval_dataloader:
        if len(all_results) % 1000 == 0:
            logger.info("Processing example: %d" % (len(all_results)))
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        with torch.no_grad():
            batch_start_logits, batch_end_logits, _ = model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask, geometric_p=args.geometric_p, window_size=args.window_size, lmb=args.lmb)
        for i, example_index in enumerate(example_indices):
            start_logits = batch_start_logits[i].detach().cpu().tolist()
            end_logits = batch_end_logits[i].detach().cpu().tolist()
            eval_feature = eval_features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            all_results.append(RawResult(unique_id=unique_id,
                                         start_logits=start_logits,
                                         end_logits=end_logits))

    preds, nbest_preds = \
        make_predictions(eval_examples, eval_features, all_results,
                         args.n_best_size, args.max_answer_length,
                         args.do_lower_case, args.verbose_logging)
    
    exact_raw, f1_raw, precision, recall, span_exact, span_f1, span_p, span_r = get_raw_scores(eval_dataset, preds, eval_examples)
    result = make_eval_dict(exact_raw, f1_raw, precision=precision, recall=recall, span_exact=span_exact, span_f1=span_f1, span_p=span_p, span_r=span_r)
    
    if verbose:
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
    return result, preds, nbest_preds

def main(args):
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

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if args.do_train:
        logger.addHandler(logging.FileHandler(os.path.join(args.output_dir, "train.log"), 'w'))
    else:
        logger.addHandler(logging.FileHandler(os.path.join(args.output_dir, "eval.log"), 'w'))
    logger.info(args)

    tokenizer = BertTokenizer.from_pretrained(
        args.model, do_lower_case=args.do_lower_case)

    if args.do_train or (not args.eval_test):
        with gzip.GzipFile(args.dev_file, 'r') as reader:
            content = reader.read().decode('utf-8').strip().split('\n')[1:]
            eval_dataset = [json.loads(line) for line in content]
        eval_examples = read_mrqa_examples(
            input_file=args.dev_file, is_training=False)
        eval_features = convert_examples_to_features(
            examples=eval_examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=False)
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
        train_examples = read_mrqa_examples(
            input_file=args.train_file, is_training=True)
        train_features = convert_examples_to_features(
                examples=train_examples,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length,
                doc_stride=args.doc_stride,
                max_query_length=args.max_query_length,
                is_training=True)

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
            for epoch in range(int(args.num_train_epochs)):
                model.train()
                logger.info("Start epoch #{} (lr = {})...".format(epoch, lr))
                for step, batch in enumerate(train_batches):
                    if n_gpu == 1:
                        batch = tuple(t.to(device) for t in batch)
                    input_ids, input_mask, segment_ids, start_positions, end_positions = batch
                    loss, _ = model(input_ids, segment_ids, input_mask, start_positions, end_positions, args.geometric_p, window_size=args.window_size, lmb=args.lmb)
                    
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

                    if (step + 1) % eval_step == 0:
                        logger.info('Epoch: {}, Step: {} / {}, used_time = {:.2f}s'.format(
                            epoch, step + 1, len(train_dataloader), time.time() - start_time))

                        save_model = False
                        if args.do_eval:
                            result, _, _ = \
                                evaluate(args, model, device, eval_dataset,
                                         eval_dataloader, eval_examples, eval_features)
                            model.train()
                            result['global_step'] = global_step
                            result['epoch'] = epoch
                            result['learning_rate'] = lr
                            result['batch_size'] = args.train_batch_size
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

    if args.do_eval:
        if args.eval_test:
            with gzip.GzipFile(args.test_file, 'r') as reader:
                content = reader.read().decode('utf-8').strip().split('\n')[1:]
                eval_dataset = [json.loads(line) for line in content]
            eval_examples = read_mrqa_examples(
                input_file=args.test_file, is_training=False)
            eval_features = convert_examples_to_features(
                examples=eval_examples,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length,
                doc_stride=args.doc_stride,
                max_query_length=args.max_query_length,
                is_training=False)
            logger.info("***** Test *****")
            logger.info("  Num orig examples = %d", len(eval_examples))
            logger.info("  Num split examples = %d", len(eval_features))
            logger.info("  Batch size = %d", args.eval_batch_size)
            all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
            all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
            eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_example_index)
            eval_dataloader = DataLoader(eval_data, batch_size=args.eval_batch_size)
        model, pretrained_weights = BLANC.from_pretrained(args.output_dir)
        if args.fp16:
            model.half()
        model.to(device)
        result, preds, nbest_preds = \
            evaluate(args, model, device, eval_dataset,
                     eval_dataloader, eval_examples, eval_features)
        with open(os.path.join(args.output_dir, PRED_FILE), "w") as writer:
            writer.write(json.dumps(preds, indent=4) + "\n")
        with open(os.path.join(args.output_dir, TEST_FILE), "w") as writer:
            for key in sorted(result.keys()):
                writer.write("%s = %s\n" % (key, str(result[key])))


if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument("--model", default=None, type=str, required=True)
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
        parser.add_argument("--train_mode", type=str, default='random_sorted', choices=['random', 'sorted', 'random_sorted'])
        parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
        parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
        parser.add_argument("--eval_test", action="store_true", help="Whether to evaluate on final test set.")
        parser.add_argument("--train_batch_size", default=32, type=int, help="Total batch size for training.")
        parser.add_argument("--eval_batch_size", default=8, type=int, help="Total batch size for predictions.")
        parser.add_argument("--learning_rate", default=None, type=float, help="The initial learning rate for Adam.")
        parser.add_argument("--num_train_epochs", default=3.0, type=float,
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
        parser.add_argument('--lmb', type=float, default=0.5)
        args = parser.parse_args()

        main(args)
