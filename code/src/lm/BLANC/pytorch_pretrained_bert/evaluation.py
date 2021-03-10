def get_raw_scores(dataset, predictions, examples):
    answers = {}
    for qa in dataset:
        #for qa in example['qas']:
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