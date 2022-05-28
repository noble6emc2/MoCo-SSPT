CUDA_VISIBLE_DEVICES=$CUDA python3 src/lm/BLANC/run_moco_baseline.py \
  --do_train \
  --model /research/d4/gds/mindahu21/bert_model/bert-base-uncased.tar.gz \
  --tokenizer /research/d4/gds/mindahu21/bert_model/bert-base-uncased-vocab.txt \
  --train_file $DATA_DIR/sspt_qg_combined_dataset.jsonl \
  --dev_file $DATA_DIR/dev.jsonl.gz \
  --test_file $DATA_DIR/test.jsonl.gz \
  --train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 1 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --eval_metric span_f1 \
  --output_dir ./checkpoints/pretrain_moco/$LABEL/$SEED \
  --eval_per_epoch 20 \
  --seed $SEED \
  --enqueue_thread_num 4 \
  --num_iteration $ITERNUM \
  --command pretraining \
  --model_type BertForQA \
  --training_lang EN
