"""
train_models.py
---------------
Standalone training script. Run:
    python train_models.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ml.pipeline import run

if __name__ == "__main__":
    run()
