import os
import sys
sys.path.append(os.getcwd())

import numpy as np
from tensorflow.keras.models import load_model

from src.visualization.visualize import evaluate_visually
import config

if __name__ == "__main__":
    print("\n[EVALUATION STAGE]\n")

    # LOAD MODEL
    model_path = os.path.join(config.MODEL_DIR, "VAE.hdf5")
    model = load_model(model_path)

    # LOAD DATA
    X_test = np.load(os.path.join(config.DATA_PREPROCESSED_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(config.DATA_PREPROCESSED_DIR, "y_test.npy"))
    n_test_samples = X_test.shape[0]

    # RESHAPE TO MODEL
    X_test = X_test.reshape(n_test_samples, config.image_shape[0], config.image_shape[0], 1)
    y_test = y_test.reshape(n_test_samples, config.image_shape[0], config.image_shape[0])

    # EVALUATE
    test_loss = round(model.evaluate(X_test, y_test),4)
    print(f"Test loss {round(test_loss, 4)}")

    # SAVE METRIC
    path = config.MODEL_TEST_METRICS
    with open(path, "a+") as f:
        f.write(f"Test {config.loss_function} = " + str(test_loss) + "\n")

    # SAVE VISUAL EVALUATION
    fig = evaluate_visually(model, X_test, y_test)
    fig.savefig(config.FIGURES_VISUAL_EVAL)
