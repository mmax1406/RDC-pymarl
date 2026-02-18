import sys
import os
import runpy

# --- Fix protobuf ---
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")

# CRITICAL: emulate `python src/main.py`
sys.path.insert(0, SRC)

sys.argv = [
    "main.py",
    "--config=pd_qmix_gru4mpe",
    "--env-config=gymma",
    "with",
    "env_args.key=pz-mpe-simple-tag-v3",
    "env_args.pretrained_wrapper=PretrainedTag",
    "t_max=100",
    "predictor_mode=none",
    "delay_aware=True",
    "cheating_start_value=1.0",
    "cheating_end_value=1.0",
    "exp_name=TAG-QMIX-B",
]

# This EXACTLY mimics: python src/main.py ...
runpy.run_path(os.path.join(SRC, "main.py"), run_name="__main__")
