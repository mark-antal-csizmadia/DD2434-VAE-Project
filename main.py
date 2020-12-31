import tensorflow as tf
import numpy as np
import matplotlib.pyplot as mlp
from random import randrange
from VAE import autoencoder, autodecoder, VAE_loss
from keras.models import Model
from keras.layers import Input

def createRandomBatch(datax, datay, num_samples=100):
    '''
        Create random batch of datax samples

        Parameters
        -----------
        datax : Tensor
                Dataset without label
        datay : numpy.ndarray
                Labels of dataset
        num_samples : int
                      number of samples contained in the batch  

        Output
        -----------
        batch_x : List<Tensors>
                  List with num_samples Tensors chosen randomly
        batch_y : List<Int>
                  List with num_samples labels corresponding to the Tensors  

    '''
    batch_x = []
    batch_y = []

    for i in range(num_samples):
        id = randrange(datax.shape[0])
        batch_x.append(datax[id])
        batch_y.append(datay[id])
    return batch_x, batch_y


if __name__ == "__main__":

    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    (training_samples, width, height) = x_train.shape
    test_samples = x_test.shape[0]

    # Plot MNIST data examples
    # for i in range(12):
    #    mlp.subplot(4, 4, 1 + i)
    #    mlp.axis('off')
    #    mlp.imshow(x_train[i], cmap='gray')

    # Normalize images (source: https://towardsdatascience.com/image-classification-in-10-minutes-with-mnist-dataset-54c35b77a38d)
    # 255 = RGB max , this will normalize it to [0,1]
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Reshape inputs
    trainX = x_train.reshape(
        (x_train.shape[0], x_train.shape[1], x_train.shape[1], 1))
    testX = x_test.reshape(
        (x_test.shape[0], x_test.shape[1], x_test.shape[1], 1))

    batchx, batchy = createRandomBatch(x_train, y_train, 100)

    input_dim = (x_train.shape[1]*x_train.shape[1],)
    hidden_dim = 512
    latent_dim = 2

    # Encoder + Decoder Model
    encoder, kl_loss = autoencoder(input_dim, hidden_dim, latent_dim)
    decoder = autodecoder(input_dim, hidden_dim, latent_dim)

    # VAE Model
    vae_input = Input(shape=input_dim)
    vae_encoder = encoder(vae_input)
    # We only need to take in consideration the last tensor
    vae_encoder = vae_encoder[-1]
    vae_decoder = decoder(vae_encoder)

    vae_loss = VAE_loss(vae_input, vae_decoder, kl_loss)
    vae = Model(vae_input, vae_decoder)

    # Getting error here
    #vae.compile(optimizer='SGD', loss=vae_loss)
    #vae.fit(x_train, batch=100)
