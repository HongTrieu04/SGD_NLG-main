#!/bin/bash

# Nếu không có biến SLURM_JOB_ID, gán giá trị mặc định
if [ -z "$SLURM_JOB_ID" ]; then
  SLURM_JOB_ID="local"
fi

model_name="mlp-prefix-t5-small-SGD"  # Gán giá trị mặc định nếu không truyền tham số
LOG=models/$model_name/logs/infer-log.$SLURM_JOB_ID
ERR=models/$model_name/logs/infer-err.$SLURM_JOB_ID

mkdir -p models/$model_name/logs

echo -e "JobID: $SLURM_JOB_ID\n======" > $LOG
echo "Time: date" >> $LOG
echo "Running on master node: hostname" >> $LOG

python infer_model.py --model_name=$model_name >> $LOG 2> $ERR

echo "Time: date" >> $LOG