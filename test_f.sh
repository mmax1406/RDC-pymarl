#!/bin/sh
#!/bin/sh

# 检查是否提供了足够的参数
if [ $# -lt 2 ]; then
    echo "Usage: $0 \"base_command\" \"log_file\""
    exit 1
fi

base_command=$1
log_file=$2

# --- AUTO-LOAD LOGIC ---
# Using sed instead of grep -oP for standard shell compatibility
EXP_NAME=$(echo "$base_command" | sed -n 's/.*exp_name="\([^"]*\)".*/\1/p')
# --- AUTO-LOAD LOGIC ---
# Using sed instead of grep -oP for standard shell compatibility
EXP_NAME=$(echo "$base_command" | sed -n 's/.*exp_name="\([^"]*\)".*/\1/p')
MODEL_ROOT="./results/models"

# Find latest weights (Standard find + sort logic)
LATEST_CKPT=$(find "$MODEL_ROOT" -maxdepth 5 -type d -name "*${EXP_NAME}*" -printf '%T+ %p\n' 2>/dev/null | sort -r | head -n 1 | cut -d' ' -f2)
# Find latest weights (Standard find + sort logic)
LATEST_CKPT=$(find "$MODEL_ROOT" -maxdepth 5 -type d -name "*${EXP_NAME}*" -printf '%T+ %p\n' 2>/dev/null | sort -r | head -n 1 | cut -d' ' -f2)

if [ -z "$LATEST_CKPT" ]; then
    echo "ERROR: No checkpoint found for $EXP_NAME"
    echo "ERROR: No checkpoint found for $EXP_NAME"
    exit 1
fi

# 循环执行 delay_value 从 0 到 12
# 循环执行 delay_value 从 0 到 12
for delay_value in $(seq 0 12); do
    echo "DEBUG: Current delay_value = $delay_value"

    # Standard shell string matching using 'case'
    case "$log_file" in
        *ss_h* | *ms_h*)
            # If log_file contains ss_h or ms_h
            if [ "$delay_value" -eq 0 ]; then
                n_expand_action=0
            else
                n_expand_action=9
            fi
            ;;
        *)
            # Default case
            n_expand_action=$delay_value
            ;;
    esac
    # Standard shell string matching using 'case'
    case "$log_file" in
        *ss_h* | *ms_h*)
            # If log_file contains ss_h or ms_h
            if [ "$delay_value" -eq 0 ]; then
                n_expand_action=0
            else
                n_expand_action=9
            fi
            ;;
        *)
            # Default case
            n_expand_action=$delay_value
            ;;
    esac

    # 构造完整命令
    full_command="${base_command} checkpoint_path=\"${LATEST_CKPT}\" evaluate=True delay_type=\"f\" delay_value=${delay_value} delay_scope=0 n_expand_action=${n_expand_action} >> ${log_file} 2>&1 &"
    full_command="${base_command} checkpoint_path=\"${LATEST_CKPT}\" evaluate=True delay_type=\"f\" delay_value=${delay_value} delay_scope=0 n_expand_action=${n_expand_action} >> ${log_file} 2>&1 &"
    
    # 执行命令
    eval "${full_command}"
    eval "${full_command}"
    
    # 等待完成
    # 等待完成
    wait
    
    # 延迟防止资源冲突
    # 延迟防止资源冲突
    sleep 2
done

echo "All commands submitted for $EXP_NAME."
