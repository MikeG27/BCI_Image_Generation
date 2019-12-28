import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR,"data")

DATA_RAW_DIR = os.path.join(DATA_DIR,"raw")
EEG_RAW_DIR = os.path.join(DATA_RAW_DIR,"EEG")
IMG_RAW_DIR = os.path.join(DATA_RAW_DIR,"IMG")

DATA_PREPROCESSED_DIR = os.path.join(DATA_DIR,"preprocessed")
EEG_PREPROCESSED_DIR = os.path.join(DATA_PREPROCESSED_DIR,"EEG")
IMG_PREPROCESSED_DIR = os.path.join(DATA_PREPROCESSED_DIR,"IMG")




