import keras
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as kb


def get_latent(gauss_params):
    '''
        Function to reparametrize mu,sigma by sampling from Gaussian

        Parameters
        -----------
        gauss_params : Tensor 
                       mu and sigma of q(z|x)

        Output
        -----------
        z : Tensor
            latent vector sampled after encoding the data
    '''

    # Parameters of q(z|x) (equation 9)
    mu, sigma = gauss_params

    # Sample epsilon and calculate latent vector Z (equation 10)
    eps = kb.random_normal(shape=(kb.shape(mu)[0], kb.shape(mu)[1]))
    z = mu + sigma * eps

    return z


def autoencoder(input_dim, hidden_dim, latent_dim=2):
    '''
        Function to perform autoencoding

        Parameters
        -----------
        input_dim : Tuple 
                    dimension of the images as (x*y,)
        hidden_dim : Integer
                    Output dimension of hidden layer
        latent_dim : Integer 
                    Output dimension of latent layer

        Output
        -----------
        encoder : Keras Model
                  NN Model of the encoding for the VAE. Contains the output of this process.

    '''
    # Layers
    input_layer = Input(shape=input_dim)
    x = Dense(hidden_dim, activation='relu')(input_layer)
    mu = Dense(latent_dim)(x)
    sigma = Dense(latent_dim)(x)

    # Reparametrization trick, pushing the sampling out as input
    # (Lambda layer used to do operations on a tensor)
    z = Lambda(get_latent, output_shape=(latent_dim,))([mu, sigma])

    encoder = Model(input_layer, [mu, sigma, z],
                    name="autoencoder")  # Create model
    # encoder.summary()

    return encoder


def autodecoder(input_dim, hidden_dim, latent_dim=2):
    '''
        Function to perform autodecoding (reconstruct latent points to original dimension)

        Parameters
        -----------
        input_dim : Tuple 
                    dimension of the images as (x*y,)
        hidden_dim : Integer
                    Output dimension of hidden layer
        latent_dim : Integer 
                    Output dimension of latent layer

        Output
        -----------
        decoder : Keras Model
                  NN Model of the decoding for the VAE. Contains the output of this process.

    '''
    # Layers
    input_layer = Input(shape=(latent_dim,))
    x = Dense(hidden_dim, activation='relu')(input_layer)
    out = Dense(input_dim[1]*input_dim[1], activation='sigmoid')(x)

    decoder = Model(input_layer, out, name="autodecoder")   # Create model
    # decoder.summary()

    return decoder
