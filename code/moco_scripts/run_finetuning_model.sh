CUDA=0 \
LABEL=trial_001_sspt_baseline_bert_finetuning \
SEED=0 \
DATA_DIR=/research/d4/gds/mindahu21/qa_dataset \
MODEL_PATH=/research/d4/gds/mindahu21/MoCo-SSPT/code/checkpoints/pretrain_moco/trial_001_baseline_sspt_loss/0 \
VOCAB_PATH=/research/d4/gds/mindahu21/MoCo-SSPT/code/checkpoints/pretrain_moco/trial_001_baseline_sspt_loss/0/vocab.txt \
bash ./moco_scripts/moco_finetuning_model.sh