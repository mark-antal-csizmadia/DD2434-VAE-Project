import tensorflow as tf
#from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as mlp
from random import randrange


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

    # Expand dimensions (2D -> 3D) for NN input
    x_train = tf.expand_dims(x_train, axis=-1)
    x_test = tf.expand_dims(x_test, axis=-1)

    batchx, batchy = createRandomBatch(x_train, y_train, 100)
