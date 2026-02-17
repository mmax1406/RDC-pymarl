#!/bin/sh

# 检查是否提供了足够的参数
if [ $# -lt 2 ]; then
    echo "Usage: $0 \"base_command\" \"log_file\""
    echo "Example: $0 'CUDA_VISIBLE_DEVICES=\"0\" nohup python ...' \"expansion_sweep.log\""
    exit 1
fi

base_command=$1
log_file=$2

# --- AUTO-LOAD LOGIC START (POSIX COMPLIANT) ---
# Using sed to extract exp_name safely in standard shell
EXP_NAME=$(echo "$base_command" | sed -n 's/.*exp_name="\([^"]*\)".*/\1/p')
MODEL_ROOT="./results/models"

# Find the latest directory. 
# Maxdepth 5 handles: models/env/algo/exp_timestamp
LATEST_CKPT=$(find "$MODEL_ROOT" -maxdepth 5 -type d -name "*${EXP_NAME}*" -printf '%T+ %p\n' 2>/dev/null | sort -r | head -n 1 | cut -d' ' -f2)

if [ -z "$LATEST_CKPT" ]; then
    echo "ERROR: No checkpoint folder found for exp_name: $EXP_NAME in $MODEL_ROOT"
    exit 1
else
    echo "DEBUG: Found latest weights at $LATEST_CKPT"
fi
# --- AUTO-LOAD LOGIC END ---

delay_value=12
# 循环执行 n_expand_action 从 1 到 12 的所有变体
# seq is usually available, but we can use a standard shell while loop for 100% compatibility
n_expand_action=1
while [ "$n_expand_action" -le 12 ]; do
    echo "DEBUG: Current n_expand_action = $n_expand_action (Delay Fixed at $delay_value)"

    # 构造完整命令
    # Injected: checkpoint_path and evaluate=True
    full_command="${base_command} checkpoint_path=\"${LATEST_CKPT}\" evaluate=True delay_type=\"f\" delay_value=${delay_value} delay_scope=0 n_expand_action=${n_expand_action} >> ${log_file} 2>&1 &"
    
    # 执行命令
    eval "${full_command}"
    
    # 等待上一个命令完成
    wait
    
    # 延迟防止资源冲突
    sleep 2
    
    # Increment counter
    n_expand_action=$((n_expand_action + 1))
done

echo "Expansion sweep for $EXP_NAME is complete."
