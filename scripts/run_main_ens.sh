#!/bin/bash
CUDA_ID="7"
TARGET_ID=7
SOURCE_ID_list="2_3_5"
Auxiliary_id_list="1_4_6"
Attack_method="GA_TDMI_fgsm"
LOSS="ce"
MAX_EPSILON=20
EPSILON=20
Num_steps=10
Momentum=1.0
batch_size=4
THRES=0.2
Intervals=5
kernel_size=5
mode="nearest"
Feature=$1

OUTPUT_DIR="Output_APR/Ensemble_TID_${TARGET_ID}_SIDLIST_${SOURCE_ID_list}_Auxiliary_id_list_${Auxiliary_id_list}_${Attack_method}_${LOSS}_thres_${THRES}_intervals_${Intervals}_steps_${Num_steps}_max_eps_${MAX_EPSILON}_eps_${EPSILON}_mu_${Momentum}_mode_${mode}_kernel_size_${kernel_size}_${Feature}"
mkdir -p ${OUTPUT_DIR}

echo ${OUTPUT_DIR}

OMP_NUM_THREADS=1 \
CUDA_VISIBLE_DEVICES=${CUDA_ID} \
python -u main_ens_attack.py \
  --save_path=${OUTPUT_DIR}/images \
  --attack_method=${Attack_method} \
  --loss_fn=${LOSS} \
  --batch_size=${batch_size} \
  --max_epsilon=${MAX_EPSILON} \
  --epsilon=${EPSILON} \
  --num_steps=${Num_steps} \
  --intervals=${Intervals} \
  --momentum=${Momentum} \
  --thres=${THRES} \
  --mode=${mode} \
  --kernel_size=${kernel_size} \
  --source_list=${SOURCE_ID_list} \
  --auxiliary_list=${Auxiliary_id_list} \
  --target_id=${TARGET_ID} \
  >> ${OUTPUT_DIR}/log.log 2>&1