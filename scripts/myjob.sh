#!/bin/bash
#SBATCH --job-name=acic_aipw
#SBATCH --output=experiments/output/acic/output_%A_%a.txt
#SBATCH --error=experiments/output/acic/error_%A_%a.txt
#SBATCH -c 15
#SBATCH -t 1:00:00
#SBATCH -p gpu_quad
#SBATCH --gres=gpu:1
#SBATCH --mem=15G
#SBATCH --array=1-10
hostname

# Add your directory to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/n/data2/hms/dbmi/beamlab/manqing/DAGawareTransformer"
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.3,max_split_size_mb:512


# python3 -m venv myenv
source myenv/bin/activate
module load gcc/9.2.0
module load python/3.10.11

# pip install -r requirements.txt

# Use the SLURM_ARRAY_TASK_ID as the sample index
sample_index=$SLURM_ARRAY_TASK_ID

python3 src/experiment.py \
        --config config/train/acic/acic_sample${SLURM_ARRAY_TASK_ID}.json \
        --dag dag_aipw \
        --estimator aipw \
        --data_name acic \
        --use_layernorm \
        --dag_attention_mask