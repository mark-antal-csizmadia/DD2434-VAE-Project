import keras
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as kb
import tensorflow as tf
from keras.utils import to_categorical


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

    # Computes KL Divergence Loss (First part of equation 24)
    # KLDivergence_Loss = -0.5 * \
    #    (kb.sum((1+kb.log(kb.square(sigma))-kb.square(mu)-kb.square(sigma)), axis=-1))
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
        KLDivergence_Loss : Tensor
            Computed Kullback Leibler Divergence on the samples

    '''
    # Layers
    input_layer = Input(shape=input_dim)
    initializer = tf.initializers.VarianceScaling(scale=2.0)
    hidden_layer = Dense(hidden_dim, activation='relu', kernel_initializer=initializer)(input_layer)
    mu = Dense(latent_dim, kernel_initializer=initializer)(hidden_layer)
    sigma = Dense(latent_dim, kernel_initializer=initializer)(hidden_layer)

    # Reparametrization trick, pushing the sampling out as input
    # (Lambda layer used to do operations on a tensor)
    z = Lambda(get_latent, output_shape=(latent_dim,))([mu, sigma])

    encoder = Model(input_layer, [mu, sigma, z],
                    name="autoencoder")  # Create model
    # encoder.summary()

    KLDivergence_Loss = -0.5 * \
        (kb.sum((1+sigma-kb.square(mu)-kb.exp(sigma)), axis=-1))

    return encoder, KLDivergence_Loss, input_layer


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
    initializer = tf.initializers.VarianceScaling(scale=2.0)
    x = Dense(hidden_dim, activation='relu', kernel_initializer=initializer)(input_layer)
    out = Dense(input_dim[0], activation='sigmoid', kernel_initializer=initializer)(x)  # Bernoulli MLP

    decoder = Model(input_layer, out, name="autodecoder")   # Create model
    # decoder.summary()

    return decoder


def VAE_loss(input_y, decoded_y, KLDivergence_Loss):
    '''
        Function to calculate the models reconstruction loss

        Parameters
        -----------
        input_y : input data y value

        decoded_y : decoded y 

        KLDivergence_Loss : Tensor
            Computed Kullback Leibler Divergence on the samples (taken from get_latent())

        Output
        -----------
        VAE_loss : 
                  The models computed reconstructed loss
    # formula for Binary Crossentropy  is taken from: https://peltarion.com/knowledge-center/documentation/modeling-view/build-an-ai-model/loss-functions/binary-crossentropy
        https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
    '''
    #Crossentropy_Loss = keras.losses.binary_crossentropy(input_y, decoded_y)
    # Crossentropy_Loss = (-1/input_y.shape[1]) * kb.sum(
    #    (input_y * kb.log(decoded_y)) + ((1-input_y)*kb.log(1-decoded_y)))
    # Crossentropy_Loss = kb.max(decoded_y, 0)-decoded_y * \
    #     input_y + kb.log(1+kb.exp((-1)*kb.abs(decoded_y)))
    # epsilon to avoid log(0) -> nan
    epsilon = 10e-15
    # New cross entropy loss yields numbers similar in magnitude to the keras binary_crossentropy below.
    # Crossentropy_Loss = tf.keras.losses.binary_crossentropy(input_y, decoded_y)
    Crossentropy_Loss = tf.reduce_mean(-input_y * kb.log(decoded_y + epsilon) - (1 - input_y) * kb.log(1 - decoded_y + epsilon))

    VAE_Loss = kb.mean(kb.mean(Crossentropy_Loss) + KLDivergence_Loss, axis=-1)
    return VAE_Loss
