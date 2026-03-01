import sys
import os
import runpy

# --- 1. GLOBALS & PATH SETUP ---
# Fix for protobuf issues in certain environments
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)

# --- 2. CONFIGURATION: TOGGLE THIS LINE ---
# Options: "tag" or "spread"
TEST_ENV = "tag" 

# --- 3. TASK MAPPING ---
# This replicates the differences between your two bash commands
TASKS = {
    "tag": {
        "env_key": "pz-mpe-simple-tag-v3",
        "exp_name": "TAG-QMIX-B",
        "extra_args": ["env_args.pretrained_wrapper=PretrainedTag"]
    },
    "spread": {
        "env_key": "pz-mpe-simple-spread-v3",
        "exp_name": "SPREAD-QMIX-B",
        "extra_args": []
    }
}

active = TASKS[TEST_ENV]

# --- 4. CONSTRUCT COMMAND LINE ARGUMENTS ---
# This mimics the 'python -u src/main.py ...' structure
sys.argv = [
    "main.py",
    "--config=pd_qmix_gru4mpe",
    "--env-config=gymma",
    "with",
    f"env_args.key={active['env_key']}",
    "t_max=1000",
    "predictor_mode=none",
    "delay_aware=True",
    "cheating_start_value=1.0",
    "cheating_end_value=1.0",
    f"exp_name={active['exp_name']}",
] + active["extra_args"]

# --- 5. EXECUTION ---
print(f"--- RUNNING {TEST_ENV.upper()} ON CPU ---")
print(f"Environment: {active['env_key']}")
print(f"Experiment:  {active['exp_name']}")
print(f"Arguments:   {' '.join(sys.argv[1:])}\n")

try:
    # This executes main.py as if it were the entry point
    runpy.run_path(os.path.join(SRC, "main.py"), run_name="__main__")
except Exception as e:
    print(f"\n[!] Execution interrupted or failed: {e}")