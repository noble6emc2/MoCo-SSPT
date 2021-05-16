import torch
import collections
import logging
import math
import time
import re
import string
from pytorch_pretrained_bert.tokenization import BasicTokenizer
from mrqa_official_eval import exact_cn_match_score, f1_score, metric_max_over_ground_truths

logger = logging.getLogger(__name__)
RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits"])


def output_loss(args, model, device, eval_dataloader, eval_features):
    for idx, (input_ids, input_mask, segment_ids, start_positions, end_positions, example_indices) in enumerate(eval_dataloader):
        all_results = []
        model.eval()
        if idx % 10 == 0:
            logger.info("Running test: %d / %d" % (idx, len(eval_dataloader)))
            
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        with torch.no_grad():
            _, overall_losses, context_losses = model(input_ids, segment_ids, input_mask, start_positions, end_positions, geometric_p=args.geometric_p, window_size=args.window_size, lmb=args.lmb)

        for i, example_index in enumerate(example_indices):
            overall_losses = overall_losses[i].detach().cpu().tolist()
            context_losses = context_losses[i].detach().cpu().tolist()
            eval_feature = eval_features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            all_results.append((unique_id, overall_losses, context_losses))

    return all_results


class MRQAEvaluator:
    @staticmethod
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
                start_indexes = MRQAEvaluator._get_best_indexes(result.start_logits, n_best_size)
                end_indexes = MRQAEvaluator._get_best_indexes(result.end_logits, n_best_size)
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
                    final_text = MRQAEvaluator.get_final_text(tok_text, orig_text, do_lower_case, verbose_logging)
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
            
            probs = MRQAEvaluator._compute_softmax(total_scores)
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

    @staticmethod
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

    @staticmethod
    def _get_best_indexes(logits, n_best_size):
        """Get the n-best logits from a list."""
        index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

        best_indexes = []
        for i in range(len(index_and_score)):
            if i >= n_best_size:
                break
            best_indexes.append(index_and_score[i][0])
        return best_indexes

    @staticmethod    
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


    @staticmethod
    def get_raw_scores(dataset, predictions, examples):
        answers = {}
        for qa in dataset:
            #for qa in example['qas']:
            answers[qa['qid']] = qa['answers']
        '''for example in dataset:
            for qa in example['qas']:
                answers[qa['qid']] = qa['answers']'''

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
                exact_cn_match_score, prediction, ground_truths)[0]
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

    @staticmethod
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

    @staticmethod
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
            MRQAEvaluator.make_predictions(eval_examples, eval_features, all_results,
                            args.n_best_size, args.max_answer_length,
                            args.do_lower_case, args.verbose_logging)
        
        exact_raw, f1_raw, precision, recall, span_exact, span_f1, span_p, span_r = MRQAEvaluator.get_raw_scores(eval_dataset, preds, eval_examples)
        result = MRQAEvaluator.make_eval_dict(exact_raw, f1_raw, precision=precision, recall=recall, span_exact=span_exact, span_f1=span_f1, span_p=span_p, span_r=span_r)
        
        if verbose:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
        return result, preds, nbest_preds


class SQuADEvaluator:
    @staticmethod
    def make_predictions(all_examples, all_features, all_results, n_best_size,
                     max_answer_length, do_lower_case, verbose_logging,
                     version_2_with_negative):
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
        scores_diff_json = collections.OrderedDict()

        for (example_index, example) in enumerate(all_examples):
            features = example_index_to_features[example_index]
            prelim_predictions = []
            score_null = 1000000
            min_null_feature_index = 0
            null_start_logit = 0
            null_end_logit = 0
            for (feature_index, feature) in enumerate(features):
                result = unique_id_to_result[feature.unique_id]
                start_indexes = SQuADEvaluator._get_best_indexes(result.start_logits, n_best_size)
                end_indexes = SQuADEvaluator._get_best_indexes(result.end_logits, n_best_size)
                if version_2_with_negative:
                    feature_null_score = result.start_logits[0] + result.end_logits[0]
                    if feature_null_score < score_null:
                        score_null = feature_null_score
                        min_null_feature_index = feature_index
                        null_start_logit = result.start_logits[0]
                        null_end_logit = result.end_logits[0]
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
            if version_2_with_negative:
                prelim_predictions.append(
                    _PrelimPrediction(
                        feature_index=min_null_feature_index,
                        start_index=0,
                        end_index=0,
                        start_logit=null_start_logit,
                        end_logit=null_end_logit))
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
                orig_doc_start = None
                orig_doc_end = None
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
                    final_text = SQuADEvaluator.get_final_text(tok_text, orig_text, do_lower_case, verbose_logging)
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

            if version_2_with_negative:
                if "" not in seen_predictions:
                    nbest.append(
                        _NbestPrediction(
                            text="",
                            start_logit=null_start_logit,
                            end_logit=null_end_logit,
                            start_index=None,
                            end_index=None))
                if len(nbest) == 1:
                    nbest.insert(0, _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0, start_index=None, end_index=None))

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
            
            probs = SQuADEvaluator._compute_softmax(total_scores)
            nbest_json = []
            for (i, entry) in enumerate(nbest):
                output = collections.OrderedDict()
                output["text"] = entry.text
                output["probability"] = probs[i]
                output["start_logit"] = entry.start_logit
                output["end_logit"] = entry.end_logit
                output["start_index"] = entry.start_index
                output["end_index"] = entry.end_index
                nbest_json.append(output)

            assert len(nbest_json) >= 1
            if not version_2_with_negative:
                all_predictions[example.qas_id] = target_entry
            else:
                score_diff = score_null - best_non_null_entry.start_logit - (
                    best_non_null_entry.end_logit)
                scores_diff_json[example.qas_id] = score_diff
                all_predictions[example.qas_id] = target_entry
            all_nbest_json[example.qas_id] = nbest_json

        return all_predictions, all_nbest_json, scores_diff_json

    @staticmethod
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

    @staticmethod
    def _get_best_indexes(logits, n_best_size):
        """Get the n-best logits from a list."""
        index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

        best_indexes = []
        for i in range(len(index_and_score)):
            if i >= n_best_size:
                break
            best_indexes.append(index_and_score[i][0])
        return best_indexes

    @staticmethod
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

    @staticmethod
    def make_qid_to_has_ans(dataset):
        qid_to_has_ans = {}
        for article in dataset:
            for p in article['paragraphs']:
                for qa in p['qas']:
                    qid_to_has_ans[qa['id']] = bool(qa['answers'])
        return qid_to_has_ans

    @staticmethod
    def normalize_answer(s):

        def remove_articles(text):
            regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
            return re.sub(regex, ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()
        return white_space_fix(remove_articles(remove_punc(lower(s))))

    @staticmethod
    def get_tokens(s):
        if not s:
            return []
        return SQuADEvaluator.normalize_answer(s).split()

    @staticmethod
    def compute_exact(a_gold, a_pred):
        return int(SQuADEvaluator.normalize_answer(a_gold) == SQuADEvaluator.normalize_answer(a_pred))

    @staticmethod
    def compute_f1(a_gold, a_pred):
        gold_toks = SQuADEvaluator.get_tokens(a_gold)
        pred_toks = SQuADEvaluator.get_tokens(a_pred)
        common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
        num_same = sum(common.values())
        if len(gold_toks) == 0 or len(pred_toks) == 0:
            return [int(gold_toks == pred_toks)] * 3
        if num_same == 0:
            return [0, 0, 0]
        precision = 1.0 * num_same / len(pred_toks)
        recall = 1.0 * num_same / len(gold_toks)
        f1 = (2 * precision * recall) / (precision + recall)
        return [precision, recall, f1]

    @staticmethod
    def get_raw_scores(dataset, preds, examples):
        exact_scores = {}
        f1_scores = {}
        scores = {}
        precision_scores = {}
        recall_scores = {}
        for article in dataset:
            for p in article['paragraphs']:
                for qa in p['qas']:
                    qid = qa['id']
                    gold_answers = [a['text'] for a in qa['answers'] if SQuADEvaluator.normalize_answer(a['text'])]
                    if not gold_answers:
                        gold_answers = ['']
                    if qid not in preds:
                        print('Missing prediction for %s' % qid)
                        continue
                    a_pred = preds[qid]["text"]
                    exact_scores[qid] = max(SQuADEvaluator.compute_exact(a, a_pred) for a in gold_answers)
                    scores[qid] = [SQuADEvaluator.compute_f1(a, a_pred) for a in gold_answers]
                    f1_scores[qid] = max([s[2] for s in scores[qid]])
                    recall_scores[qid] = max([s[1] for s in scores[qid]])
                    precision_scores[qid] = max([s[0] for s in scores[qid]])
        
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
        
        def select_g(sgs, egs):
            n = len(sgs)
            si = min([i for i in sgs])
            ei = max([i for i in egs])
            i2n = [0] * (ei + 1)
            for i in range(si, ei + 1):
                for j in range(n):
                    i2n[i] += 1 if sgs[j] <= i and i <= egs[j] else 0
            m = max(i2n)
            st = 0; et = 0
            for i in range(si, ei + 1):
                if i2n[i] == m:
                    st = i
                    break
            for i in range(ei, si - 1, -1):
                if i2n[i] == m:
                    et = i
                    break
            return st, et

        span_f1 = {}
        span_exact = {}
        span_precision = {}
        span_recall = {}
        for example in examples:
            qid = example.qas_id
            sgs = example.start_positions
            egs = example.end_positions
            
            sf = preds[qid]["start_index"]
            ef = preds[qid]["end_index"]
            if sf == None:
                sf = -1
            if ef == None:
                ef = -1
            
            n_can = len(sgs)
            span_exact[qid] = 0.0
            for j in range(n_can):
                if sf == sgs[j] and ef == egs[j]:
                    span_exact[qid] = 1.0
                    break
            span_f1[qid] = max([get_f1(sf, ef, sgs[i], egs[i]) for i in range(n_can)])
            span_precision[qid] = max([get_precision(sf, ef, sgs[i], egs[i]) for i in range(n_can)])
            span_recall[qid] = max([get_recall(sf, ef, sgs[i], egs[i]) for i in range(n_can)])
                
        return exact_scores, f1_scores, precision_scores, recall_scores, span_exact, span_f1, span_precision, span_recall

    @staticmethod
    def apply_no_ans_threshold(scores, na_probs, qid_to_has_ans, na_prob_thresh):
        new_scores = {}
        for qid, s in scores.items():
            pred_na = na_probs[qid] > na_prob_thresh
            if pred_na:
                new_scores[qid] = float(not qid_to_has_ans[qid])
            else:
                new_scores[qid] = s
        return new_scores

    @staticmethod
    def make_eval_dict(exact_scores, f1_scores, p_scores={}, r_scores={}, span_exact={}, span_f1={}, span_p={}, span_r={}, qid_list=None):
        if not qid_list:
            total = len(exact_scores)
            return collections.OrderedDict([
                ('exact', 100.0 * sum(exact_scores.values()) / total),
                ('f1', 100.0 * sum(f1_scores.values()) / total),
                ('precision', 100.0 * sum(p_scores.values()) / total),
                ('recall', 100.0 * sum(r_scores.values()) / total),
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
                ('precision', 100.0 * sum(p_scores.values()) / total),
                ('recall', 100.0 * sum(r_scores.values()) / total),
                ('span_exact', 100.0 * sum(span_exact.values()) / total),
                ('span_f1', 100.0 * sum(span_f1.values()) / total),
                ('span_precision', 100.0 * sum(span_p.values()) / total),
                ('span_recall', 100.0 * sum(span_r.values()) / total),
                ('total', total),
            ])

    @staticmethod
    def merge_eval(main_eval, new_eval, prefix):
        for k in new_eval:
            main_eval['%s_%s' % (prefix, k)] = new_eval[k]

    @staticmethod
    def find_best_thresh(preds, scores, na_probs, qid_to_has_ans):
        num_no_ans = sum(1 for k in qid_to_has_ans if not qid_to_has_ans[k])
        cur_score = num_no_ans
        best_score = cur_score
        best_thresh = 0.0
        qid_list = sorted(na_probs, key=lambda k: na_probs[k])
        for i, qid in enumerate(qid_list):
            if qid not in scores:
                continue
            if qid_to_has_ans[qid]:
                diff = scores[qid]
            else:
                if preds[qid]:
                    diff = -1
                else:
                    diff = 0
            cur_score += diff
            if cur_score > best_score:
                best_score = cur_score
                best_thresh = na_probs[qid]
        return 100.0 * best_score / len(scores), best_thresh

    @staticmethod
    def find_all_best_thresh(main_eval, preds, exact_raw, f1_raw, na_probs, qid_to_has_ans):
        best_exact, exact_thresh = SQuADEvaluator.find_best_thresh(preds, exact_raw, na_probs, qid_to_has_ans)
        best_f1, f1_thresh = SQuADEvaluator.find_best_thresh(preds, f1_raw, na_probs, qid_to_has_ans)
        main_eval['best_exact'] = best_exact
        main_eval['best_exact_thresh'] = exact_thresh
        main_eval['best_f1'] = best_f1
        main_eval['best_f1_thresh'] = f1_thresh

    @staticmethod
    def evaluate(args, model, device, eval_dataset, eval_dataloader,
             eval_examples, eval_features, na_prob_thresh=1.0, pred_only=False):
        all_results = []
        model.eval()
        eval_time_s = time.time()
        for idx, (input_ids, input_mask, segment_ids, example_indices) in enumerate(eval_dataloader):
            if idx % 10 == 0:
                logger.info("Running test: %d / %d" % (idx, len(eval_dataloader)))
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            with torch.no_grad():
                batch_start_logits, batch_end_logits, _ = model(input_ids, segment_ids, input_mask, geometric_p=args.geometric_p, window_size=args.window_size, lmb=args.lmb)
            for i, example_index in enumerate(example_indices):
                start_logits = batch_start_logits[i].detach().cpu().tolist()
                end_logits = batch_end_logits[i].detach().cpu().tolist()
                eval_feature = eval_features[example_index.item()]
                unique_id = int(eval_feature.unique_id)
                all_results.append(RawResult(unique_id=unique_id,
                                            start_logits=start_logits,
                                            end_logits=end_logits))
        eval_time_e = time.time()

        preds, nbest_preds, na_probs = \
            SQuADEvaluator.make_predictions(eval_examples, eval_features, all_results,
                            args.n_best_size, args.max_answer_length,
                            args.do_lower_case, args.verbose_logging,
                            args.version_2_with_negative)
        
        if pred_only:
            if args.version_2_with_negative:
                for k in preds:
                    if na_probs[k] > na_prob_thresh:
                        preds[k] = ''
            return {}, preds, nbest_preds

        if args.version_2_with_negative:
            qid_to_has_ans = SQuADEvaluator.make_qid_to_has_ans(eval_dataset)
            has_ans_qids = [k for k, v in qid_to_has_ans.items() if v]
            no_ans_qids = [k for k, v in qid_to_has_ans.items() if not v]
            exact_raw, f1_raw, _, _, span_exact, span_f1, span_p, span_r = SQuADEvaluator.get_raw_scores(eval_dataset, preds, eval_examples)
            exact_thresh = SQuADEvaluator.apply_no_ans_threshold(exact_raw, na_probs, qid_to_has_ans, na_prob_thresh)
            f1_thresh = SQuADEvaluator.apply_no_ans_threshold(f1_raw, na_probs, qid_to_has_ans, na_prob_thresh)
            result = SQuADEvaluator.make_eval_dict(exact_thresh, f1_thresh)
            if has_ans_qids:
                has_ans_eval = SQuADEvaluator.make_eval_dict(exact_thresh, f1_thresh, qid_list=has_ans_qids)
                SQuADEvaluator.merge_eval(result, has_ans_eval, 'HasAns')
            if no_ans_qids:
                no_ans_eval = SQuADEvaluator.make_eval_dict(exact_thresh, f1_thresh, qid_list=no_ans_qids)
                SQuADEvaluator.merge_eval(result, no_ans_eval, 'NoAns')
            SQuADEvaluator.find_all_best_thresh(result, preds, exact_raw, f1_raw, na_probs, qid_to_has_ans)
            for k in preds:
                if na_probs[k] > result['best_f1_thresh']:
                    preds[k] = {}
        else:
            exact_raw, f1_raw, p_raw, r_raw, span_exact, span_f1, span_p, span_r = SQuADEvaluator.get_raw_scores(eval_dataset, preds, eval_examples)
            result = SQuADEvaluator.make_eval_dict(exact_raw, f1_raw, p_scores=p_raw, r_scores=r_raw, span_exact=span_exact, span_f1=span_f1, span_p=span_p, span_r=span_r)
        
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
        logger.info("Eval time: {:.06f}".format(eval_time_e - eval_time_s))
        return result, preds, nbest_preds
