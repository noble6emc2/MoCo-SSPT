PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/lm/BLANC/run_mrqa_blanc_pretraining_chinese.py \
  --do_train \
  --model /home/mindahu/bert-base-chinese.tar.gz \
  --tokenizer /home/mindahu/bert-base-chinese-vocab.txt \
  --train_file $DATA_DIR/all.json \
  --dev_file $DATA_DIR/dev.jsonl.gz \
  --test_file $DATA_DIR/test.jsonl.gz \
  --train_batch_size 8 \
  --eval_batch_size 16  \
  --learning_rate 2e-5 \
  --num_train_epochs 1 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --eval_metric span_f1 \
  --output_dir ./checkpoints/pertraining_cn_cotraining/$LABEL/$SEED \
  --eval_per_epoch 20 \
  --seed $SEED \
  --geometric_p $GEOP \
  --window_size $WINS \
  --lmb $LMB \
  --is_co_training true \
  --co_training_mode moving_loss\
  --enqueue_thread_num 2 \
  --num_iteration 37500 \
  --moving_loss_num 8 \
  --theta 0.8
