"""
Implementation of the Deep Embedded Self-Organizing Map model
Autoencoder helper function

@author Florent Forest
@version 2.0
"""

from keras.models import Model
from keras.layers import Input, Dense
from keras.layers import Conv2D, Flatten, Conv2DTranspose, Reshape, Activation, LeakyReLU
from keras import backend as K
import numpy as np


def cnn_autoencoder1(input_dim, z_dim):
    
 
    im_size = int(np.sqrt(input_dim))
    x = Input(shape=(input_dim,), name='encoder_input')
    encoded = x
    encoded = Reshape((im_size, im_size, 1), name='encoder_0')(x)
    encoded = Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu', name='encoder_1')(encoded)
    encoded = Conv2D(filters=64, kernel_size=3, strides=2, padding='same', activation='relu', name='encoder_2')(encoded)
    encoded = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu', name='encoder_3')(encoded)
    encoded = Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu', name='encoder_4')(encoded)
    shape_before_flattening = K.int_shape(encoded)[1:]
    encoded = Flatten(name='encoder_5')(encoded)
    encoded = Dense(z_dim, name='encoder_6')(encoded)
    decoded = encoded


    decoded = Dense(np.prod(shape_before_flattening), name='decoder_6')(decoded)
    decoded = Reshape(shape_before_flattening, name='decoder_5')(decoded)
    decoded = Conv2DTranspose(filters=32, kernel_size=3, strides=1, padding = 'same', activation='relu', 
                              name='decoder_4')(decoded)
    decoded = Conv2DTranspose(filters=64, kernel_size=3, strides=1, padding = 'same', activation='relu', 
                              name='decoder_3')(decoded)
    decoded = Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding = 'same', activation='relu', 
                              name='decoder_2')(decoded)
    decoded = Conv2DTranspose(filters=1, kernel_size=3, strides=1, padding = 'same', activation='sigmoid', 
                              name='decoder_1')(decoded)
    decoded = Flatten(name='decoder_0')(decoded)


    autoencoder = Model(inputs=x, outputs=decoded, name='AE')
    encoder = Model(inputs=x, outputs=encoded, name='encoder')

    encoded_input = Input(shape=(z_dim,))
    decoded = encoded_input
    for i in range(7-1, -1, -1):
        decoded = autoencoder.get_layer('decoder_%d' % i)(decoded)
    # Decoder model
    decoder = Model(inputs=encoded_input, outputs=decoded, name='decoder')
    
    return autoencoder, encoder, decoder

def cnn_autoencoder(input_dim, z_dim):
    
 
    im_size = int(np.sqrt(input_dim))
    x = Input(shape=(input_dim,), name='encoder_input')
    encoded = x
    encoded = Reshape((im_size, im_size, 1), name='encoder_0')(x)
    encoded = Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu', name='encoder_1')(encoded)
    encoded = Conv2D(filters=64, kernel_size=3, strides=2, padding='same', activation='relu', name='encoder_2')(encoded)
    encoded = Conv2D(filters=64, kernel_size=3, strides=2, padding='same', activation='relu', name='encoder_3')(encoded)
    encoded = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu', name='encoder_4')(encoded)
    shape_before_flattening = K.int_shape(encoded)[1:]
    encoded = Flatten(name='encoder_5')(encoded)
    encoded = Dense(z_dim, name='encoder_6')(encoded)
    decoded = encoded


    decoded = Dense(np.prod(shape_before_flattening), name='decoder_6')(decoded)
    decoded = Reshape(shape_before_flattening, name='decoder_5')(decoded)
    decoded = Conv2DTranspose(filters=64, kernel_size=3, strides=1, padding = 'same', activation='relu', 
                              name='decoder_4')(decoded)
    decoded = Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding = 'same', activation='relu', 
                              name='decoder_3')(decoded)
    decoded = Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding = 'same', activation='relu', 
                              name='decoder_2')(decoded)
    decoded = Conv2DTranspose(filters=1, kernel_size=3, strides=1, padding = 'same', activation='sigmoid', 
                              name='decoder_1')(decoded)
    decoded = Flatten(name='decoder_0')(decoded)


    autoencoder = Model(inputs=x, outputs=decoded, name='AE')
    encoder = Model(inputs=x, outputs=encoded, name='encoder')

    encoded_input = Input(shape=(z_dim,))
    decoded = encoded_input
    for i in range(7-1, -1, -1):
        decoded = autoencoder.get_layer('decoder_%d' % i)(decoded)
    # Decoder model
    decoder = Model(inputs=encoded_input, outputs=decoded, name='decoder')
    
    return autoencoder, encoder, decoder



def mlp_autoencoder(encoder_dims, act='relu', init='glorot_uniform'):
    """
    Fully connected symmetric autoencoder model.

    # Arguments
        encoder_dims: list of number of units in each layer of encoder. encoder_dims[0] is input dim, encoder_dims[-1] is units in hidden layer (latent dim).
        The decoder is symmetric with encoder, so number of layers of the AE is 2*len(encoder_dims)-1
        act: activation of AE intermediate layers, not applied to Input, Hidden and Output layers
        init: initialization of AE layers
    # Return
        (ae_model, encoder_model, decoder_model): AE, encoder and decoder models
    """
    n_stacks = len(encoder_dims) - 1

    # Input
    x = Input(shape=(encoder_dims[0],), name='input')
    # Internal layers in encoder
    encoded = x
    for i in range(n_stacks-1):
        encoded = Dense(encoder_dims[i + 1], activation=act, kernel_initializer=init, name='encoder_%d' % i)(encoded)
    # Hidden layer (latent space)
    encoded = Dense(encoder_dims[-1], kernel_initializer=init, name='encoder_%d' % (n_stacks - 1))(encoded) # hidden layer, latent representation is extracted from here
    # Internal layers in decoder
    decoded = encoded
    for i in range(n_stacks-1, 0, -1):
        decoded = Dense(encoder_dims[i], activation=act, kernel_initializer=init, name='decoder_%d' % i)(decoded)
    # Output
    decoded = Dense(encoder_dims[0], kernel_initializer=init, name='decoder_0')(decoded)

    # AE model
    autoencoder = Model(inputs=x, outputs=decoded, name='AE')

    # Encoder model
    encoder = Model(inputs=x, outputs=encoded, name='encoder')

    # Create input for decoder model
    encoded_input = Input(shape=(encoder_dims[-1],))
    # Internal layers in decoder
    decoded = encoded_input
    for i in range(n_stacks-1, -1, -1):
        decoded = autoencoder.get_layer('decoder_%d' % i)(decoded)
    # Decoder model
    decoder = Model(inputs=encoded_input, outputs=decoded, name='decoder')

    return autoencoder, encoder, decoder