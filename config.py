import os


# Preprocessing

image_shape = (30, 30)

train_ratio=0.75
test_ratio=0.15
validation_ratio=0.10

# Training
intermediate_dim = 512
latent_dim = 2

epochs = 1000
batch_size = 128
loss_function = "mean_squared_error"

#######################################################

# Roots
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")

# RAW
RAW_DIR = os.path.join(DATA_DIR, "raw")
RAW_IMG_DIR = os.path.join(RAW_DIR, "IMG")
RAW_EEG_DIR = os.path.join(RAW_DIR, "EEG")

# DATASET

GDRIVE_ID = '1nvSTPTUv5r6bc7axUJcz6dKHUjsqdgjm'
GDRIVE_FILE = "mnist-64s.csv"

# PREPROCESSED
DATA_PREPROCESSED_DIR = os.path.join(DATA_DIR, "preprocessed")
PREPROCESSED_IMG_DIR = os.path.join(DATA_PREPROCESSED_DIR, "IMG")
PREPROCESSED_EEG_DIR = os.path.join(DATA_PREPROCESSED_DIR, "EEG")

# MODELS
MODEL_DIR = os.path.join(ROOT_DIR, "models")
MODEL_TENSORBOARD_DIR = os.path.join(MODEL_DIR, "logs/fit/")
MODEL_CHECKPOINTER_DIR = os.path.join(MODEL_DIR, "VAE.hdf5")
MODEL_TRAIN_METRICS = os.path.join(MODEL_DIR, "train_metrics.txt")
MODEL_TEST_METRICS = os.path.join(MODEL_DIR, "test_metrics.txt")


# NOTEBOOKS
NOTEBOOKS_DIR = os.path.join(ROOT_DIR, "notebooks")

# REFERENCES
REFERENCES_DIR = os.path.join(ROOT_DIR, "references")

# REPORTS
REPORTS_DIR = os.path.join(ROOT_DIR,"reports")

# FIGURES
FIGURES_DIR = os.path.join(REPORTS_DIR,"figures")
FIGURES_LEARNING_CURVE = os.path.join(FIGURES_DIR,"learning_curve.png")
FIGURES_VISUAL_EVAL = os.path.join(FIGURES_DIR,"visual_eval.png")
