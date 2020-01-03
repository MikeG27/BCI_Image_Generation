import os


# Roots
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR,"data")

# RAW
DATA_RAW_DIR = os.path.join(DATA_DIR,"raw")

# RAW IMG
RAW_IMG_DIR = os.path.join(DATA_RAW_DIR, "IMG")
RAW_IMG_TRAIN_DIR = os.path.join(RAW_IMG_DIR, "train")
RAW_IMG_TEST_DIR = os.path.join(RAW_IMG_DIR, "test")

# RAW EEG
EEG_RAW_DIR = os.path.join(DATA_RAW_DIR,"EEG")

# PREPROCESSED
DATA_PREPROCESSED_DIR = os.path.join(DATA_DIR,"preprocessed")

# PREPROCESSED IMG
PREPROCESSED_IMG_DIR = os.path.join(DATA_PREPROCESSED_DIR, "IMG")
PREPROCESSED_IMG_TRAIN_DIR = os.path.join(PREPROCESSED_IMG_DIR, "train")
PREPROCESSED_IMG_TEST_DIR = os.path.join(PREPROCESSED_IMG_DIR, "test")

#PREPROCESSED EEG
PREPROCESSED_EEG_DIR = os.path.join(DATA_PREPROCESSED_DIR, "EEG")




