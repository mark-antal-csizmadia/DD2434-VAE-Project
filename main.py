import tensorflow as tf
import numpy as np
import matplotlib.pyplot as mlp
from random import randrange
from VAE import autoencoder, autodecoder, VAE_loss
from keras.models import Model
from keras.layers import Input
from keras.utils import plot_model
from keras.optimizers import Adam


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

    # Reshape data to (60000,784) in mnist case
    xTRAIN = np.reshape(x_train, [-1, x_train.shape[1]*x_train.shape[1]])
    xTEST = np.reshape(x_test, [-1, x_test.shape[1]*x_test.shape[1]])

    # Normalize
    xTRAIN = xTRAIN.astype('float32') / 255
    xTEST = xTEST.astype('float32') / 255
    
    input_dim = (x_train.shape[1]*x_train.shape[1],)
    hidden_dim = 512
    latent_dim = 2

    # Encoder + Decoder Model
    encoder, kl_loss, vae_input = autoencoder(
        input_dim, hidden_dim, latent_dim)
    decoder = autodecoder(input_dim, hidden_dim, latent_dim)

    # VAE Model
    vae_encoder = encoder(vae_input)
    # We only need to take in consideration the last tensor
    vae_encoder = vae_encoder[-1]
    vae_decoder = decoder(vae_encoder)

    vae_loss = VAE_loss(vae_input, vae_decoder, kl_loss)
    vae = Model(vae_input, vae_decoder)
    # Save .png of model. Uncomment if not needed.
    #plot_model(vae, to_file='vae.png', show_shapes=True)

    vae.add_loss(vae_loss)
    optimizer = Adam(learning_rate=1e-3)
    vae.compile(optimizer=optimizer)
    # Always do minibatches that are a power of 2 and try to be around 16-64
    # https://twitter.com/ylecun/status/989610208497360896?lang=en
    vae.fit(xTRAIN, xTRAIN, epochs=10, batch_size=32, shuffle=True)

    # Reconstruct training data.
    xTRAIN_re = vae.predict(xTRAIN)

    # Plot first plot_n image. idx in the loop will always start at zero.
    plot_n = 20
    fig, axs = mlp.subplots(plot_n, 2)
    for idx, re in enumerate(xTRAIN_re[:plot_n]):
        axs[idx, 0].imshow(re.reshape(28, 28))
        axs[idx, 1].imshow(xTRAIN[idx].reshape(28, 28))
    mlp.show()
