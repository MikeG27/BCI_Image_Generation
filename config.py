import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR,"data")

DATA_RAW_DIR = os.path.join(DATA_DIR,"raw")

IMG_RAW_DIR = os.path.join(DATA_RAW_DIR,"IMG")
IMG_TRAIN_RAW_DIR = os.path.join(IMG_RAW_DIR,"train")
IMG_TEST_RAW_DIR = os.path.join(IMG_RAW_DIR,"test")

EEG_RAW_DIR = os.path.join(DATA_RAW_DIR,"EEG")

DATA_PREPROCESSED_DIR = os.path.join(DATA_DIR,"preprocessed")
IMG_PREPROCESSED_DIR = os.path.join(DATA_PREPROCESSED_DIR,"IMG")
IMG_TRAIN_PREPROCESSED_DIR = os.path.join(IMG_PREPROCESSED_DIR,"train")
IMG_TEST_PREPROCESSED_DIR = os.path.join(IMG_PREPROCESSED_DIR,"test")

EEG_PREPROCESSED_DIR = os.path.join(DATA_PREPROCESSED_DIR,"EEG")


# Zrobic metode do buildowania wszystkich folderow


