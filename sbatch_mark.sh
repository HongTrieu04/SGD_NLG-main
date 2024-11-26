#!/bin/bash

# Gán giá trị mặc định nếu không chạy trên SLURM
if [ -z "$SLURM_JOB_ID" ]; then
  SLURM_JOB_ID="local"
fi

# Tên mô hình và file cần đánh giá
model_name=${model_name:-"mlp-prefix-t5-small-SGD"}  # Mặc định nếu không truyền
filename=${filename:-"test_cases.txt"}              # File cần đánh giá

# Đường dẫn log
LOG=models/$model_name/logs/mark-log.$SLURM_JOB_ID
ERR=models/$model_name/logs/mark-err.$SLURM_JOB_ID

# Tạo thư mục logs nếu chưa tồn tại
mkdir -p models/$model_name/logs

echo -e "JobID: $SLURM_JOB_ID\n======" > $LOG
echo "Time: date" >> $LOG
echo "Running on local machine: hostname" >> $LOG

# Chạy script đánh giá
python mark.py --file=$filename >> $LOG 2> $ERR

echo "Time: date" >> $LOG