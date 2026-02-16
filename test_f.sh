#!/bin/bash

# 检查是否提供了足够的参数
if [ $# -lt 2 ]; then
    echo "Usage: $0 \"base_command\" \"log_file\""
    echo "Example: $0 'CUDA_VISIBLE_DEVICES=\"0\" nohup python -u src/main.py ...' \"tag_test_qmix_b\""
    exit 1
fi

base_command=$1
log_file=$2

# --- NEW: AUTO-LOAD LOGIC START ---
# 1. Extract exp_name from the base_command string
# This looks for exp_name="ANYTHING" and grabs ANYTHING
EXP_NAME=$(echo "$base_command" | grep -oP 'exp_name="\K[^"]+')

# 2. Define where your models are stored (standard PyMARL/EPYMRL path)
MODEL_ROOT="./results/models"

# 3. Find the latest directory matching that experiment name
# We look for directories containing the EXP_NAME and sort by modification time (newest first)
LATEST_CKPT=$(find "$MODEL_ROOT" -maxdepth 3 -type d -name "*${EXP_NAME}*" -printf '%T+ %p\n' 2>/dev/null | sort -r | head -n 1 | cut -d' ' -f2)

if [ -z "$LATEST_CKPT" ]; then
    echo "ERROR: No checkpoint folder found for exp_name: $EXP_NAME in $MODEL_ROOT"
    exit 1
else
    echo "DEBUG: Found latest weights at $LATEST_CKPT"
fi
# --- AUTO-LOAD LOGIC END ---

# 循环执行delay_value从0到12的所有变体
for delay_value in $(seq 0 12); do
    echo "DEBUG: Current delay_value = $delay_value"

    # 根据log_file和delay_value设置n_expand_action的值
    if [[ "$log_file" == *"ss_h"* ]]; then
        n_expand_action=$([ $delay_value -eq 0 ] && echo 0 || echo 9)
    elif [[ "$log_file" == *"ms_h"* ]]; then
        n_expand_action=$([ $delay_value -eq 0 ] && echo 0 || echo 9)
    else
        n_expand_action=$delay_value
    fi

    # 构造完整命令
    # Added: checkpoint_path="${LATEST_CKPT}"
    full_command="${base_command} checkpoint_path=\"${LATEST_CKPT}\" delay_type=\"f\" delay_value=${delay_value} delay_scope=0 n_expand_action=${n_expand_action} >> ${log_file} 2>&1 &"
    
    # 执行命令
    eval ${full_command}
    
    # 等待上一个命令完成
    wait
    
    # 可选：添加一些延迟以防止资源冲突
    sleep 2
done

echo "All commands for $EXP_NAME have been completed."