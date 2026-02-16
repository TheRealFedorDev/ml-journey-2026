import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.absolute()

DATA_DIR = BASE_DIR / 'data' / 'raw'
TRAIN_DIR = DATA_DIR / 'train'
VAL_DIR = DATA_DIR / 'val'
TEST_DIR = DATA_DIR / 'test'

MODELS_DIR = BASE_DIR / 'models'
MODELS_DIR.mkdir(exist_ok=True)