#!/bin/sh

# 检查是否提供了足够的参数
if [ $# -lt 2 ]; then
    echo "Usage: $0 \"base_command\" \"log_file\""
    echo "Example: $0 'CUDA_VISIBLE_DEVICES=\"0\" nohup python ...' \"tag_test_uf.log\""
    exit 1
fi

base_command=$1
log_file=$2

# --- AUTO-LOAD LOGIC START (POSIX COMPLIANT) ---
# Using sed instead of grep -oP for better shell compatibility
EXP_NAME=$(echo "$base_command" | sed -n 's/.*exp_name="\([^"]*\)".*/\1/p')
MODEL_ROOT="./results/models"

# Find the latest directory matching the experiment name
# Bumping maxdepth to 5 to handle your environment/algorithm subfolder structure
LATEST_CKPT=$(find "$MODEL_ROOT" -maxdepth 5 -type d -name "*${EXP_NAME}*" -printf '%T+ %p\n' 2>/dev/null | sort -r | head -n 1 | cut -d' ' -f2)

if [ -z "$LATEST_CKPT" ]; then
    echo "ERROR: No checkpoint folder found for exp_name: $EXP_NAME in $MODEL_ROOT"
    exit 1
else
    echo "DEBUG: Found latest weights at $LATEST_CKPT"
fi
# --- AUTO-LOAD LOGIC END ---

# 分为集内 (In-distribution) 和集外 (Out-of-distribution)
for i in 0 1; do
    echo "DEBUG: Current iteration i = $i"
    
    if [ "$i" -eq 0 ]; then
        delay_value=6
        delay_scope=3
    else
        delay_value=9
        delay_scope=3
    fi

    # Standard shell string matching using 'case' instead of [[ ... ]]
    case "$log_file" in
        *ss_h* | *ms_h*)
            # History-based models use a fixed expansion (9 for MPE)
            n_expand_action=9
            ;;
        *)
            # Standard models expand to max possible delay: delay + scope
            n_expand_action=$((delay_value + delay_scope))
            ;;
    esac

    # 构造完整命令
    # Added: checkpoint_path and evaluate=True
    full_command="${base_command} checkpoint_path=\"${LATEST_CKPT}\" evaluate=True delay_type=\"uf\" delay_value=${delay_value} delay_scope=${delay_scope} n_expand_action=${n_expand_action} >> ${log_file} 2>&1 &"
    
    # 执行命令 (Use quotes for safety)
    eval "${full_command}"
    
    # 等待上一个命令完成
    wait
    
    # 延迟防止资源冲突
    sleep 2
done

echo "Unfixed delay tests for $EXP_NAME submitted successfully."
