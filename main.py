import tensorflow as tf
import matplotlib.pyplot as mlp

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

    # Expand dimensions (2D -> 3D) for NN input
    x_train = tf.expand_dims(x_train, axis=-1)
    x_test = tf.expand_dims(x_test, axis=-1)

    # Normalize images (source: https://towardsdatascience.com/image-classification-in-10-minutes-with-mnist-dataset-54c35b77a38d)
    x_train = x_train.astype('float32')
    # 255 = RGB max , this will normalize it to [0,1]
    x_train = x_train / 255.0
    x_test = x_test.astype('float32')
    x_test = x_test / 255.0
