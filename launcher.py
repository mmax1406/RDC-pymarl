import sys
import os
import runpy

# --- Fix protobuf ---
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")

testEnv = "spread"  # "tag" or "spread"
delay = 10

# --- 1. CONFIGURATION: TOGGLE THIS LINE ---
if testEnv == "tag":
    ENV_KEY = "pz-mpe-simple-tag-v3"
elif testEnv == "spread":   
    ENV_KEY = "pz-mpe-simple-spread-v3"  # Switch to "pz-mpe-simple-spread-v3" as needed

# --- 2. HARD-CODED PATH MAPPING ---
PATHS = {
    "pz-mpe-simple-tag-v3": {
        "checkpoint": os.path.join(ROOT, "results/models/pz-mpe-simple-tag-v3/pd_qmix/TAG-QMIX-B_2026-02-12 23:50:33.406275_pd_qmix_seed707788613"),
        "exp_name": "TAG-QMIX-B"
    },
    "pz-mpe-simple-spread-v3": {
        "checkpoint": os.path.join(ROOT, "results/models/pz-mpe-simple-spread-v3/pd_qmix/SPREAD-QMIX-B_2026-02-14 10:16:02.677269_pd_qmix_seed419326129"),
        "exp_name": "SPREAD-QMIX-B"
    }
}

# Extract the values based on the toggle
checkpoint = PATHS[ENV_KEY]["checkpoint"]
experiment_name = PATHS[ENV_KEY]["exp_name"]

print(f"\n--- HARD-CODED EXECUTION ---")
print(f"Env: {ENV_KEY}")
print(f"Path: {checkpoint}\n")

# --- 3. RUNNER CONFIGURATION ---
print(f"\n=== Running with delay = {delay} ===")
sys.path.insert(0, SRC)

if ENV_KEY == "pz-mpe-simple-tag-v3":
    sys.argv = [
        "main.py",
        "--config=pd_qmix_gru4mpe",
        "--env-config=gymma",
        "with",
        f"env_args.key={ENV_KEY}",
        "env_args.pretrained_wrapper=PretrainedTag",
        "predictor_mode=none",
        f"exp_name={experiment_name}",
        "test_nepisode=100",
        f"checkpoint_path={checkpoint}",
        "evaluate=True",
        "delay_type=f",
        f"delay_value={delay}",
        "delay_scope=0",
        f"n_expand_action={delay}",
    ]

else:
    sys.argv = [
        "main.py",
        "--config=pd_qmix_gru4mpe",
        "--env-config=gymma",
        "with",
        f"env_args.key={ENV_KEY}",
        "predictor_mode=none",
        f"exp_name={experiment_name}",
        "test_nepisode=2",
        f"checkpoint_path={checkpoint}",
        "evaluate=True",
        "delay_type=f",
        f"delay_value={delay}",
        "delay_scope=0",
        f"n_expand_action={delay}",
    ]

# Run the simulation
runpy.run_path(os.path.join(SRC, "main.py"), run_name="__main__")