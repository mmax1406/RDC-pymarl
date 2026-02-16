#!/bin/bash

# 检查是否提供了足够的参数
if [ $# -lt 2 ]; then
    echo "Usage: $0 \"base_command\" \"log_file\""
    echo "Example: $0 'CUDA_VISIBLE_DEVICES=\"0\" nohup python -u src/main.py ...' \"tag_test_qmix_b\""
    exit 1
fi

base_command=$1
log_file=$2

# --- AUTO-LOAD LOGIC START ---
# Extract exp_name from the base_command
EXP_NAME=$(echo "$base_command" | grep -oP 'exp_name="\K[^"]+')
MODEL_ROOT="./results/models"

# Find the latest directory matching that experiment name
# We sort by modification time (newest first) to pick the most recent training run
LATEST_CKPT=$(find "$MODEL_ROOT" -maxdepth 3 -type d -name "*${EXP_NAME}*" -printf '%T+ %p\n' 2>/dev/null | sort -r | head -n 1 | cut -d' ' -f2)

if [ -z "$LATEST_CKPT" ]; then
    echo "ERROR: No checkpoint folder found for exp_name: $EXP_NAME in $MODEL_ROOT"
    exit 1
else
    echo "DEBUG: Found latest weights at $LATEST_CKPT"
fi
# --- AUTO-LOAD LOGIC END ---

# 分为集内 (In-distribution) 和集外 (Out-of-distribution)
for i in $(seq 0 1); do
    echo "DEBUG: Current iteration i = $i"
    
    if [ $i -eq 0 ]; then
        delay_value=6
        delay_scope=3
    else
        delay_value=9
        delay_scope=3
    fi

    # 根据 log_file 和 delay_value 设置 n_expand_action 的值
    if [[ "$log_file" == *"ss_h"* ]] || [[ "$log_file" == *"ms_h"* ]]; then
        # History-based models use a fixed expansion (usually 9 for MPE)
        n_expand_action=9
    else
        # Standard models expand to cover the maximum possible delay in the range
        n_expand_action=$((delay_value + delay_scope))
    fi

    # 构造完整命令
    # Added: checkpoint_path="${LATEST_CKPT}" and ensure evaluate=True is present
    full_command="${base_command} checkpoint_path=\"${LATEST_CKPT}\" evaluate=True delay_type=\"uf\" delay_value=${delay_value} delay_scope=${delay_scope} n_expand_action=${n_expand_action} >> ${log_file} 2>&1 &"
    
    # 执行命令
    eval ${full_command}
    
    # 等待上一个命令完成
    wait
    
    # 延迟防止资源冲突
    sleep 2
done

echo "Unfixed delay tests for $EXP_NAME submitted successfully."