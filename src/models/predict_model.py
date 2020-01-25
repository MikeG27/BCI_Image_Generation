import os
import sys

import numpy as np
from tensorflow.keras.models import load_model

sys.path.append(os.getcwd())
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
    test_loss = model.evaluate(X_test, y_test)
    print(f"Test loss {round(test_loss, 4)}")

    # SAVE METRIC
    filename = "metrics.txt"
    path = os.path.join(config.MODEL_DIR, filename)

    with open(path, "a+") as f:
        f.write("Test Loss = " + str(test_loss) + "\n")
