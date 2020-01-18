import os
import numpy as np
import argparse

# TODO : ogarnąc syspath dla argparesów

from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.utils import plot_model
from keras import backend as K
from keras.optimizers import RMSprop
from keras.metrics import binary_crossentropy

# Custom

import config
from config import intermediate_dim, batch_size, latent_dim,epochs
from config import FIGURES_DIR
from src.models.callbacks import tensorboard,checkpoint,reduce_lr
from src.visualization.visualize import plot_learning_curve


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', type=int, default=epochs,
                        help='number of epochs')
    parser.add_argument('-b', type=int, default=batch_size,
                        help="batch_size")

    args = parser.parse_args()

    return args

def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)
    # Returns:
        z (tensor): sampled latent vector
    """
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]  # Returns the shape of tensor or variable as a tuple of int or None entries.
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def encoder_model(inputs):
    x = Dense(intermediate_dim, activation='relu')(inputs)  # 512 neuronow

    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)
    # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, name='z')([z_mean, z_log_var])
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    return encoder, z_mean, z_log_var


def decoder_model():
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = Dense(intermediate_dim, activation='relu')(latent_inputs)
    outputs = Dense(original_dim[0], activation='sigmoid')(x)

    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')
    return decoder

def vae_model(encoder, decoder):
    outputs = decoder(encoder(inputs)[2])  # biore outputy z dekodera
    vae = Model(inputs, outputs, name='vae')  # tworzę VAE
    return vae

def vae_loss(y_true, y_pred):
    xent_loss = binary_crossentropy(K.flatten(y_true), K.flatten(y_pred))
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    vae_loss = K.mean(xent_loss + kl_loss)
    return vae_loss


if __name__ == "__main__":

    #Parse Data
    args = parser()
    batch_size = args.b
    epochs = args.e

    #Load data

    X_train = np.load(os.path.join(config.DATA_PREPROCESSED_DIR, "X_train.npy"))
    X_test = np.load(os.path.join(config.DATA_PREPROCESSED_DIR, "X_test.npy"))
    X_valid = np.load(os.path.join(config.DATA_PREPROCESSED_DIR, "X_valid.npy"))

    y_train = np.load(os.path.join(config.DATA_PREPROCESSED_DIR, "y_train.npy"))
    y_test = np.load(os.path.join(config.DATA_PREPROCESSED_DIR, "y_test.npy"))
    y_valid = np.load(os.path.join(config.DATA_PREPROCESSED_DIR, "y_valid.npy"))

    input_shape = X_train[0].shape
    original_dim = X_train[0].shape

    # Check shapes

    print("Shapes info : \n")

    print(f"X_train shape : {X_train.shape}")
    print(f"X_test shape : {X_test.shape}")
    print(f"X_valid shape : {X_valid.shape}")

    print(f"y_train shape : {y_train.shape}")
    print(f"y_test shape : {y_test.shape}")
    print(f"X_valid shape : {y_valid.shape}")

    # Initialize encoder
    inputs = Input(shape=input_shape, name='encoder_input')
    encoder, z_mean, z_log_var = encoder_model(inputs)
    encoder.summary()

    # Initialize decoder
    decoder = decoder_model()
    decoder.summary()

    # Initialize VAE
    vae = vae_model(encoder, decoder)
    vae.summary()

    # Compile model
    optimizer = RMSprop(lr=0.000007, rho=0.9, decay=0.0)
    loss = vae_loss
    vae.compile(optimizer=optimizer, loss=loss)

    # Train VAE
    history = vae.fit(x=X_train, y=y_train, epochs=epochs, batch_size=batch_size,
            callbacks=[tensorboard, checkpoint, reduce_lr])


    # Save learning curve
    fig = plot_learning_curve(history)
    filepath = os.path.join(FIGURES_DIR,"learning_curve.png")
    fig.savefig(filepath)
