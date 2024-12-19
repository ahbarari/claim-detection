#!/bin/bash
#SBATCH --partition=batch
#SBATCH --time=12:00:00
#SBATCH --gpus=4
#SBATCH --ntasks=1


module load Miniforge3
source ~/.bashrc
conda_env="thesisenv"
conda activate $conda_env

pretrained_model=microsoft/deberta-base
model_name=deberta-base

# pretrained_model=roberta-base
# model_name=roberta-base

train_batch=16
num_epochs=100

train_dataset_dir=thesis-codebase/data/
val_dataset_dir=thesis-codebase/data/
test_dataset_dir=thesis-codebase/data/

cuda_device=-1
learning_rate=1e-3
random_seed=42 

test_claim=cw_diabetes

task_names=clef,nlp4if
train_claims=clef,nlp4if


echo random seed ${random_seed}
echo learning_rate ${learning_rate}

task_adapter_paths=thesis-codebase/models/${model_name}_clef_${random_seed}_${learning_rate}_${train_batch}_adapter,thesis-codebase/models/${model_name}_nlp4if_${random_seed}_${learning_rate}_${train_batch}_adapter

model_id=${model_name}_${task_names}_${random_seed}_${learning_rate}_${train_batch}_adapter_fusion
predictions_dir=thesis-codebase/data/results/${model_id}/${test_claim}
model_path=thesis-codebase/models/${model_id}

mkdir -p $predictions_dir

echo $model_id
echo Train claim $train_claims
echo Test claim $test_claim


python /home/s6ambara/thesis-codebase/fusion_eval.py \
--pretrained_model $pretrained_model \
--random_seed $random_seed \
--model_path $model_path \
--test_dataset_dir $test_dataset_dir \
--attention_scores_output $predictions_dir \
--test_claim $test_claim \
--task_adapter_paths ${task_adapter_paths} \


conda deactivate