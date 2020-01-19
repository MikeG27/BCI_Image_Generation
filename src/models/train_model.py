import argparse
import os
# path
import sys

import numpy as np
from tensorflow.keras.layers import Input, Dense, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Reshape
from tensorflow.keras.layers import Dense, Activation


sys.path.append(os.getcwd())

import config
from config import batch_size, epochs
from config import FIGURES_DIR
from src.models.callbacks import tensorboard, checkpoint, reduce_lr
from src.visualization.visualize import plot_learning_curve


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', type=int, default=epochs,
                        help='number of epochs')
    parser.add_argument('-b', type=int, default=batch_size,
                        help="batch_size")

    args = parser.parse_args()

    return args


def build_vae(input_shape):

    width = input_shape[0]
    height = input_shape[1]
    channels = input_shape[2]

    # encoder
    input_img = Input(batch_shape=(None,width,height,channels))
    squeeze = Conv2D(256, 3, 3, padding='same', input_shape=input_shape)(input_img)
    squeeze = Activation('relu')(squeeze)
    squeeze = BatchNormalization()(squeeze)
    squeeze = MaxPooling2D(pool_size=(2, 2))(squeeze)
    squeeze = UpSampling2D(size=(2, 2))(squeeze)

    # bottleneck
    squeeze = Conv2D(128, 3, 3, kernel_initializer='glorot_uniform')(squeeze)

    squeeze0 = Activation('relu')(squeeze)
    squeeze0 = Dense(100)(squeeze0)
    squeeze0 = Reshape((30, 30))(squeeze0)
    squeeze0 = Activation('relu')(squeeze0)

    model = Model(inputs=input_img, outputs=squeeze0)

    return model

def save_traning_metrics(filename,history):
    metrics_loss = round(history.history["loss"][-1], 4)
    metrics_val_loss = round(history.history["val_loss"][-1], 4)
    with open(filename, "w+") as f:
        f.write("Loss = " + str(metrics_loss) + "\n")
        f.write("Val Loss = " + str(metrics_val_loss) + "\n")


if __name__ == "__main__":
    # Parse Data
    args = parser()
    batch_size = args.b
    epochs = args.e

    # LOAD DATA

    X_train = np.load(os.path.join(config.DATA_PREPROCESSED_DIR, "X_train.npy"))
    X_valid = np.load(os.path.join(config.DATA_PREPROCESSED_DIR, "X_valid.npy"))

    y_train = np.load(os.path.join(config.DATA_PREPROCESSED_DIR, "y_train.npy"))
    y_valid = np.load(os.path.join(config.DATA_PREPROCESSED_DIR, "y_valid.npy"))

    # Check shapes

    print("Shapes info : \n")

    print(f"X_train shape : {X_train.shape}")
    print(f"X_valid shape : {X_valid.shape}")

    print(f"y_train shape : {y_train.shape}")
    print(f"X_valid shape : {y_valid.shape}")


    # GET MODEL

    input_shape = (30, 30, 1)
    vae = build_vae(input_shape)
    vae.summary()

    # RESHAPE DATA TO MODEL

    n_train_samples = X_train.shape[0]
    n_valid_samples = X_valid.shape[0]

    X_train = X_train.reshape(n_train_samples, input_shape[1], input_shape[1], 1)
    y_train = y_train.reshape(n_train_samples, input_shape[1], input_shape[1])
    X_valid = X_valid.reshape(n_valid_samples, input_shape[1], input_shape[1], 1)
    y_valid = y_valid.reshape(n_valid_samples, input_shape[1], input_shape[1])

    # SET OPTIMIZER AND LOSS

    optimizer = optimizers.RMSprop(lr=0.0005, rho=0.9, epsilon=None, decay=0.0)
    loss = 'mean_squared_error'
    vae.compile(loss=loss, optimizer=optimizer)

    # TRAIN VAE
    history = vae.fit(X_train, y_train, verbose=1,
                      epochs=epochs, validation_data=(X_valid, y_valid),
                      batch_size=batch_size, callbacks=[tensorboard, checkpoint, reduce_lr])

    # SAVE LEARNING CURVE
    print("Save learning curve as image...")
    fig = plot_learning_curve(history)
    filepath = os.path.join(FIGURES_DIR, "learning_curve.png")
    fig.savefig(filepath)


    # SAVE LEARNING METRICS

    print("Save learning metrics..")
    filename = "metrics.txt"
    path = os.path.join(config.MODEL_DIR, filename)
    save_traning_metrics(path,history)
