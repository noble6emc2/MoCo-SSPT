#!/bin/bash
#SBATCH --job-name=M_BASELINE
#SBATCH --mail-user=1155177603@link.cuhk.edu.hk
#SBATCH --mail-type=ALL
#SBATCH --output=/research/d4/gds/mindahu21/MoCo-SSPT/code/moco_scripts/output_moco_baseline.txt ##Do not use "~" point to your home!
#SBATCH --gres=gpu:4
cd /research/d4/gds/mindahu21/MoCo-SSPT/code
bash ./moco_scripts/run_baseline.sh