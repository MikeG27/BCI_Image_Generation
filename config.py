import os


# Roots
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR,"data")

# RAW
RAW_DIR = os.path.join(DATA_DIR,"raw")
RAW_IMG_DIR = os.path.join(RAW_DIR, "IMG")
RAW_EEG_DIR = os.path.join(RAW_DIR,"EEG")

# PREPROCESSED
DATA_PREPROCESSED_DIR = os.path.join(DATA_DIR,"preprocessed")
PREPROCESSED_IMG_DIR = os.path.join(DATA_PREPROCESSED_DIR, "IMG")
PREPROCESSED_EEG_DIR = os.path.join(DATA_PREPROCESSED_DIR, "EEG")




