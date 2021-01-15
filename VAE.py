""" Contributors: Diogo Pinheiro, Jakob Lindén, Márk Csizmadia, Patrick Jonsson
"""

from keras.layers import Dense, Lambda
from keras.models import Model, load_model
from keras import backend as kb
from keras.utils import plot_model
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import datetime
import scipy.io
import yaml
import os

# Global constants.
# Epsilon: avoid numerical instability issues such as log(0) -> nan.
EPSILON = 1e-6
tf.random.set_seed(221)


class VAE_EGDB(Model):
    """ Variational Auto-Encoder class via subclassing keras.Model.
    Encoder is Gaussian -> q_{phi}(z|x) = N(z; mu, exp(log_var))
    Decoder is Bernoulli -> p_{theta}(x|z) = Bern(p) where p technically a probability value corresponding to each
    value in the input_dim elements of the output Tensor. See Appendix C
    Contributed: Diogo Pinheiro, Jakob Lindén, Márk Csizmadia, Patrick Jonsson
    """

    def __init__(self, input_dim, encoder_hidden_dim, latent_dim, decoder_hidden_dim, name):
        """ Init function of the VAE class.
        Contributed: Jakob Lindén, Patrick Jonsson

        Parameters
        ----------
        input_dim : int
            The number of input dimensions, that is, the number of nodes in the first layer of the encoder and the
            last, output layer of the decoder.

        encoder_hidden_dim : int
            The number of nodes in the hidden layer of the encoder.

        latent_dim : int
            The number of latent dimensions into which the VAE encodes the input data.

        decoder_hidden_dim : int
             The number of nodes in the hidden layer of the decoder.

        name : str
            The name of the model.
        """
        # Inherit everything from keras.Model.
        super(VAE_EGDB, self).__init__(name=name)

        # Paper: All parameters, both variational and generative, were initialized by random sampling from N (0, 0.01)
        self.initializer = tf.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
        # Tanh activation functions in MLP and paper says MLP (not ReLU in MLP)
        self.activation = "tanh"

        # Create the VAE layers.
        self.encoder_hidden = Dense(
            units=encoder_hidden_dim, activation=self.activation, kernel_initializer=self.initializer)

        self.mu = Dense(units=latent_dim, kernel_initializer=self.initializer, name="mu")

        self.log_var = Dense(units=latent_dim, kernel_initializer=self.initializer, name="log_var")

        self.z = Lambda(self.get_latent, output_shape=(latent_dim,), name="z")

        self.decoder_hidden = Dense(
            units=decoder_hidden_dim, activation=self.activation, kernel_initializer=self.initializer)

        # The decoder is a Bernoulli MLP. Has binary cross-entropy loss, see later.
        self.reconstruction = \
            Dense(units=input_dim, activation='sigmoid', kernel_initializer=self.initializer, name="reconstruction")

    def get_latent(self, gauss_params):
        """ Function to re-parametrize mu,log_var by sampling from Gaussian
        Contributed: Diogo Pinheiro

        Parameters
        -----------
        gauss_params : tf.Tensor
           mu and log_var of q(z|x), both have shape (batch_size, latent_dim) where batch_size is the
            mini-batch size of the optimizer and latent_dim is an argument to the class constructor denoting the
            number of latent dimensions

        Returns
        -----------
        z : tf.Tensor
            latent vector sampled after encoding the data. Has shape (batch_size, latent_dim) where batch_size is the
            mini-batch size of the optimizer and latent_dim is an argument to the class constructor denoting the
            number of latent dimensions
        """
        # Parameters of q(z|x) (equation 9)
        mu, log_var = gauss_params

        # Sample epsilon and calculate latent vector Z (equation 10)
        eps = kb.random_normal(shape=(kb.shape(mu)[0], kb.shape(mu)[1]))
        # standard deviation = sqrt(sigma) ; using the exponential assures that the result is positive
        std = kb.exp(0.5*log_var)
        z = mu + std * eps
        return z

    def encode(self, inputs):
        """ Encoder forward-propagation. Encodes input data into a latent distribution.
        Contributed: Patrick Jonsson

        Parameters
        -----------
        inputs : tf.Tensor
            Input data. Has shape (batch_size, input_dim) where batch_size is the mini-batch size of the optimizer and
            input_dim is the number of input nodes in the first layer of the encoder network

        Returns
        -----------
        mu : tf.Tensor
            Expectation of encoded distribution (Gaussian so expectation is the mean) as vector.
            Has shape (batch_size, latent_dim) where batch_size is the mini-batch size of the optimizer and
            latent_dim is an argument to the class constructor denoting the number of latent dimensions

        log_var : tf.Tensor
            The log of the variance of encoded distribution as vector.
            Has shape (batch_size, latent_dim) where batch_size is the mini-batch size of the optimizer and
            latent_dim is an argument to the class constructor denoting the number of latent dimensions

        z : tf.Tensor
            latent vector sampled after encoding the data. Has shape (batch_size, latent_dim) where batch_size is the
            mini-batch size of the optimizer and latent_dim is an argument to the class constructor denoting the
            number of latent dimensions
        """
        # encoder_hidden shape=(batch_size, encoder_hidden_dim)
        encoder_hidden = self.encoder_hidden(inputs)
        # mu shape=(batch_size, latent_dim)
        mu = self.mu(encoder_hidden)
        # log_var shape=(batch_size, latent_dim)
        log_var = self.log_var(encoder_hidden)
        # z shape=(batch_size, latent_dim)
        z = self.z([mu, log_var])
        return mu, log_var, z

    def decode(self, z):
        """ Decoder forward-propagation. Decodes the latent distribution into the reconstruction of the input data.
        This is the generative aspect of the VAE. Therefore, this function can be used to generate data from the
        learned latent distribution.
        The decoder of VAE_EGDB is Bernoulli -> p_{theta}(x|z) = Bern(p) where p is technically a probability value
        corresponding to each value in the input_dim elements of the output Tensor. See Appendix C
        Contributed: Márk Csizmadia

        Parameters
        -----------
        z : tf.Tensor
            latent vector sampled after encoding the data. Has shape (batch_size, latent_dim) where batch_size is the
            mini-batch size of the optimizer and latent_dim is an argument to the class constructor denoting the
            number of latent dimensions

        Returns
        -----------
        reconstruction : tf.Tensor
            The reconstruction of the input data. The decoder decodes the latent distribution encoded by the encoder,
            Has shape (batch_size, input_dim) where batch_size is the mini-batch size of the optimizer and
            input_dim is the number of input nodes in the first layer of the encoder network

        None
            Make sure different VAEs have the same number of elements returned be the decode function.
            For plotting later.
        """
        # decoder_hidden shape=(batch_size, decoder_hidden_dim)
        decoder_hidden = self.decoder_hidden(z)
        # reconstruction shape=(batch_size, input_dim)
        reconstruction = self.reconstruction(decoder_hidden)
        return reconstruction, None

    def call(self, inputs):
        """ Overrides the call function of keras.Model via subclassing. Gets called at each optimization step.
        Contributed: Jakob Lindén, Diogo Pinheiro

        Parameters
        ----------
        inputs : tf.Tensor
            Input data. Has shape (batch_size, input_dim) where batch_size is the mini-batch size of the optimizer and
            input_dim is the number of input nodes in the first layer of the encoder network
        Returns
        -------
        reconstruction : tf.Tensor
            Reconstruction of the input data. Has shape (batch_size, input_dim) where batch_size is the mini-batch
            size of the optimizer and input_dim is the number of input nodes in the first layer of the encoder network

        """
        # Attach model parts together (i.e.: create computation graph).
        mu, log_var, z = self.encode(inputs)
        reconstruction, _ = self.decode(z)

        # Create the loss tensors.
        # Computes negative! KL Divergence Loss (First part of equation 24, but negative here to have -KLD)
        negative_kl_loss = kb.mean(
            0.5 * kb.sum(-log_var + kb.exp(log_var) + kb.square(mu) - 1, axis=1))

        # Computes the Binary Cross-Entropy Loss that is equivalent to the negative binary loglikelihood
        # (comes from the sigmoid activation function of the Bernoulli MLP layer in the decoder)
        negative_log_likelihood = \
            kb.mean(-kb.sum(inputs * kb.log(reconstruction + EPSILON) +
            (1 - inputs) * kb.log(1 - reconstruction + EPSILON), axis=1))

        # Create the overall VAE loss.
        # ELBO in the paper is ELBO = KLD + Log_likelihood, ELBO in the paper is maximized
        # We minimize the negative ELBO: - ELBO = - KLD - Log_likelihood = - KLD + BCE (in the Bernoulli case)
        vae_loss = negative_log_likelihood + negative_kl_loss
        self.add_loss(vae_loss)

        # In the case of the Bernoulli output layer, reconstruction is the expectation of the Bernoulli distribution
        # which is p, the probability parameter of the distribution.
        # This is what we return as the reconstruction values in the output vector.
        # Return the reconstructed observations.
        return reconstruction


class VAE_EGDG(Model):
    """ Variational Auto-Encoder class via subclassing keras.Model.
    Encoder is Gaussian -> q_{phi}(z|x) = N(z; mu, exp(log_var))
    Decoder is Gaussian -> p_{theta}(x|z) = N(x; reconstruction_mu, exp(reconstruction_log_var)
    where reconstruction_mu is the mean of the distribution in the Gaussian output layer (coming from one layer)
    and exp(reconstruction_log_var) is the variance of the distribution in the Gaussian output layer
    (coming from another layer). See Appendix C.
    Contributed: Diogo Pinheiro, Jakob Lindén, Márk Csizmadia, Patrick Jonsson
    """
    def __init__(self, input_dim, encoder_hidden_dim, latent_dim, decoder_hidden_dim, name):
        """ Init function of the VAE class.
        Contributed: Jakob Lindén, Patrick Jonsson

        Parameters
        ----------
        input_dim : int
            The number of input dimensions, that is, the number of nodes in the first layer of the encoder and the
            last, output layer of the decoder.

        encoder_hidden_dim : int
            The number of nodes in the hidden layer of the encoder.

        latent_dim : int
            The number of latent dimensions into which the VAE encodes the input data.

        decoder_hidden_dim : int
             The number of nodes in the hidden layer of the decoder.

        name : str
            The name of the model.
        """
        # Inherit everything from keras.Model.
        super(VAE_EGDG, self).__init__(name=name)
        # Paper: All parameters, both variational and generative, were initialized by random sampling from N (0, 0.01)
        self.initializer = tf.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
        # Tanh activation functions in MLP and paper says MLP (not ReLU in MLP)
        self.activation = "tanh"

        # Create the VAE layers.
        self.encoder_hidden = Dense(
            units=encoder_hidden_dim, activation=self.activation, kernel_initializer=self.initializer)

        self.mu = Dense(units=latent_dim, kernel_initializer=self.initializer, name="mu")

        self.log_var = Dense(units=latent_dim, kernel_initializer=self.initializer, name="log_var")

        self.z = Lambda(self.get_latent, output_shape=(latent_dim,), name="z")

        self.decoder_hidden = Dense(
            units=decoder_hidden_dim, activation=self.activation, kernel_initializer=self.initializer)

        # The decoder is a Gaussian MLP, has Gaussian log-likelihood loss, see later.
        self.reconstruction_mu = \
            Dense(units=input_dim, activation='sigmoid', kernel_initializer=self.initializer, name="reconstruction_mu")

        log_var_clip_val = 50
        self.reconstruction_log_var = \
            Dense(units=input_dim, activation=lambda v: kb.clip(v, -log_var_clip_val, log_var_clip_val),
                  kernel_initializer=self.initializer, name="reconstruction_log_var")

    def get_latent(self, gauss_params):
        """ Function to re-parametrize mu,log_var by sampling from Gaussian
        Contributed: Diogo Pinheiro

        Parameters
        -----------
        gauss_params : tf.Tensor
           mu and log_var of q(z|x), both have shape (batch_size, latent_dim) where batch_size is the
            mini-batch size of the optimizer and latent_dim is an argument to the class constructor denoting the
            number of latent dimensions

        Returns
        -----------
        z : tf.Tensor
            latent vector sampled after encoding the data. Has shape (batch_size, latent_dim) where batch_size is the
            mini-batch size of the optimizer and latent_dim is an argument to the class constructor denoting the
            number of latent dimensions
        """
        # Parameters of q(z|x) (equation 9)
        mu, log_var = gauss_params

        # Sample epsilon and calculate latent vector Z (equation 10)
        eps = kb.random_normal(shape=(kb.shape(mu)[0], kb.shape(mu)[1]))
        # standard deviation = sqrt(sigma) ; using the exponential assures that the result is positive
        std = kb.exp(0.5*log_var)
        z = mu + std * eps
        return z

    def encode(self, inputs):
        """ Encoder forward-propagation. Encodes input data into a latent distribution.
        Contributed: Patrick Jonsson

        Parameters
        -----------
        inputs : tf.Tensor
            Input data. Has shape (batch_size, input_dim) where batch_size is the mini-batch size of the optimizer and
            input_dim is the number of input nodes in the first layer of the encoder network

        Returns
        -----------
        mu : tf.Tensor
            Expectation of encoded distribution (Gaussian so expectation is the mean) as vector.
            Has shape (batch_size, latent_dim) where batch_size is the mini-batch size of the optimizer and
            latent_dim is an argument to the class constructor denoting the number of latent dimensions

        log_var : tf.Tensor
            The log of the variance of encoded distribution as vector.
            Has shape (batch_size, latent_dim) where batch_size is the mini-batch size of the optimizer and
            latent_dim is an argument to the class constructor denoting the number of latent dimensions

        z : tf.Tensor
            latent vector sampled after encoding the data. Has shape (batch_size, latent_dim) where batch_size is the
            mini-batch size of the optimizer and latent_dim is an argument to the class constructor denoting the
            number of latent dimensions
        """
        # encoder_hidden shape=(batch_size, encoder_hidden_dim)
        encoder_hidden = self.encoder_hidden(inputs)
        # mu shape=(batch_size, latent_dim)
        mu = self.mu(encoder_hidden)
        # log_var shape=(batch_size, latent_dim)
        log_var = self.log_var(encoder_hidden)
        # z shape=(batch_size, latent_dim)
        z = self.z([mu, log_var])
        return mu, log_var, z

    def decode(self, z):
        """ Decoder forward-propagation. Decodes the latent distribution into the reconstruction of the input data.
        This is the generative aspect of the VAE. Therefore, this function can be used to generate data from the
        learned latent distribution.
        The decoder of VAE_EGDG is Gaussian -> p_{theta}(x|z) = N(x; reconstruction_mu, exp(reconstruction_log_var)
        where reconstruction_mu is the mean of the distribution in the Gaussian output layer (coming from one layer)
        and exp(reconstruction_log_var) is the variance of the distribution in the Gaussian output layer
        (coming from another layer). See Appendix C.
        Contributed: Márk Csizmadia

        Parameters
        -----------
        z : tf.Tensor
            latent vector sampled after encoding the data. Has shape (batch_size, latent_dim) where batch_size is the
            mini-batch size of the optimizer and latent_dim is an argument to the class constructor denoting the
            number of latent dimensions

        Returns
        -----------
        reconstruction_mu : tf.Tensor
            The mean of the distribution in the Gaussian output layer (coming from one layer).
            The decoder decodes the latent distribution encoded by the encoder,
            Has shape (batch_size, input_dim) where batch_size is the mini-batch size of the optimizer and
            input_dim is the number of input nodes in the first layer of the encoder network

        reconstruction_log_var : tf.Tensor
            exp(reconstruction_log_var) is the variance of the distribution in the Gaussian output layer
            (coming from another layer).
            The decoder decodes the latent distribution encoded by the encoder,
            Has shape (batch_size, input_dim) where batch_size is the mini-batch size of the optimizer and
            input_dim is the number of input nodes in the first layer of the encoder network
        """
        # decoder_hidden shape=(batch_size, decoder_hidden_dim)
        decoder_hidden = self.decoder_hidden(z)
        # reconstruction_mu shape=(batch_size, input_dim)
        reconstruction_mu = self.reconstruction_mu(decoder_hidden)
        # reconstruction_log_var shape=(batch_size, input_dim)
        reconstruction_log_var = self.reconstruction_log_var(decoder_hidden)
        return reconstruction_mu, reconstruction_log_var

    def call(self, inputs):
        """ Overrides the call function of keras.Model via subclassing. Gets called at each optimization step.
        Contributed: Diogo Pinheiro, Jakob Lindén, Márk Csizmadia, Patrick Jonsson

        Parameters
        ----------
        inputs : tf.Tensor
            Input data. Has shape (batch_size, input_dim) where batch_size is the mini-batch size of the optimizer and
            input_dim is the number of input nodes in the first layer of the encoder network
        Returns
        -------
        reconstruction : tf.Tensor
            Reconstruction of the input data. Has shape (batch_size, input_dim) where batch_size is the mini-batch
            size of the optimizer and input_dim is the number of input nodes in the first layer of the encoder network

        """
        # Attach model parts together (i.e.: create computation graph).
        mu, log_var, z = self.encode(inputs)
        reconstruction_mu, reconstruction_log_var = self.decode(z)

        # Create the loss tensors.
        # Computes negative! KL Divergence Loss (First part of equation 24, but negative here to have -KLD)
        negative_kl_loss = kb.mean(
            0.5 * kb.sum(-log_var + kb.exp(log_var) + kb.square(mu) - 1, axis=1))

        # Compute negative log-likelihood of normal distribution. MSE does not seem to work!
        #negative_mse = kb.mean(-kb.sum(kb.square(inputs - reconstruction_mu), axis=1))
        
        x_prec = kb.exp(-reconstruction_log_var)
        x_diff = inputs - reconstruction_mu
        x_power = -0.5 * kb.square(x_diff) * x_prec
        negative_log_likelihood = \
            kb.mean(-kb.sum(-0.5 * (reconstruction_log_var + np.log(2 * np.pi)) + x_power, axis=1))
        
        #negative_log_likelihood = negative_mse

        # Create the overall VAE loss.
        # ELBO in the paper is ELBO = KLD + Log_likelihood, ELBO in the paper is maximized
        # We minimize the negative ELBO: - ELBO = - KLD - Log_likelihood (Log_likelihood is the log-likelihood of Gauss)
        vae_loss = negative_log_likelihood + negative_kl_loss
        self.add_loss(vae_loss)

        # In the case of the Gaussian output layer, reconstruction_mu is the expectation of the Gaussian distribution.
        # This is what we return as the reconstruction values in the output vector.
        return reconstruction_mu


def plot_imgs_compare(n_imgs, x, y, x_reconstructed, fig_name):
    """ Plots the reconstructed images vs. their ground-truth counterparts.
    Contributed: Diogo Pinheiro, Jakob Lindén, Márk Csizmadia

    Parameters
    ----------
    n_imgs : int
        The number of images to be shown in the plot.
    x : tf.Tensor
        The input data of shape (n_data, height, width) where n_data is the number of images, height is the height of
        the image in pixels, and width is the width of the image of pixels.
        CHANGE THIS AND ADD AN RGB DIMENSION FOR THE ALGORITHM TO BE ABLE WORK WITH RGB DATA.

    x_reconstructed : tf.Tensor
        The reconstructed data of shape (n_data, height, width) where n_data is the number of images, height is the
        height of the image in pixels, and width is the width of the image of pixels.
        CHANGE THIS AND ADD AN RGB DIMENSION FOR THE ALGORITHM TO BE ABLE WORK WITH RGB DATA.

    y : tf.Tensor
        The ground-truth label of the input data of shape (n_data,) where n_data is the number of images.

    save_img : bool
        Whether to save the generate figure or not.

    Returns
    -------
    None

    """
    # Extract n_imgs input observations and reconstructions at random indices.
    n_imgs_all = x.shape[0]
    idx_imgs = np.random.choice(n_imgs_all, n_imgs)
    x_show = x[idx_imgs]
    # If there are labels, use them. No labels in the case of frey.
    if y is not None:
        y_show = y[idx_imgs]
    x_reconstructed_show = x_reconstructed[idx_imgs]

    # Create figure.
    fig_height = n_imgs*2
    fig_width = 2
    gs1 = gridspec.GridSpec(n_imgs, 2)
    gs1.update(wspace=0.005, hspace=0.05)  # set the spacing between axes.
    fig, axs = plt.subplots(n_imgs, 2, figsize=(fig_width, fig_height))

    # Plot the subplots of the figure.
    for n_idx, _ in enumerate(idx_imgs):
        ax_left = axs[n_idx, 0]
        ax_right = axs[n_idx, 1]
        ax_left.imshow(x_reconstructed_show[n_idx])
        ax_right.imshow(x_show[n_idx])
        ax_left.set_title(f"Reconstr.")
        # If there are labels, use them. No labels in the case of frey.
        if y is not None:
            ax_right.set_title(f"Input (l: {y_show[n_idx]})")
        else:
            ax_right.set_title("Input")
        ax_left.axis('off')
        ax_right.axis('off')

    # Tight layout.
    #fig.tight_layout()

    # Save image.
    plt.savefig(os.path.join("images", fig_name))

    plt.show()


def plot_lowerbound(history, neg_elbo_values, dataset_name, latent_dim, x_axis_label, x_axis_scale, n_data, epochs,
                    ylim, fig_name):
    """ Plot likelihood lower bound. The loss curves of training and validation.
    Contributed: Diogo Pinheiro, Jakob Lindén, Márk Csizmadia, Patrick Jonsson

    Parameters
    ----------
    history : History
        Training and validation history.

    neg_elbo_values : bool
        If True, the ELBO values (loss values) are plotted as in the paper, that is, as negative values.
        This is because in the paper the ELBO is being maximized. We minimize the negative ELBO, which is equivalent
        from an optimization point of view.

    dataset_name : str
        The name of the data set. mnist for MNIST, and frey for Frey Face.

    latent_dim : int
            The number of latent dimensions into which the VAE encodes the input data.

    x_axis_label : str
        The label on the x-axis. In the paper, they show the number of samples. samples for # Training samples evaluated
        and epochs for # Epochs evaluated.

    x_axis_scale : str
        The scale of the x-axis. In the paper they do log of number of samples. linear for linear and log for log.

    n_data : int
        The number of samples in the training data set.

    epochs : int
        The number of epochs used in the training process.

    ylim : tuple
        y_min and y_max, in this order for limit the y-axis values. In the paper, the y-axis values are limited.
        If None, no limit is applied. Note that the limit may not show the area of interest in the plot.

    Returns
    -------
    None

    """
    # Retrieve losses.
    training_losses = history.history['loss']
    validation_losses = history.history['val_loss']

    # If neg_elbo_values is True, the ELBO is plotted as in the paper, that is,
    # negative values (ELBO is being maximized).
    
    if neg_elbo_values:
        training_losses = [-training_loss for training_loss in training_losses]
        validation_losses = [-validation_loss for validation_loss in validation_losses]
    else:
        pass
    

    # Select label on x-axis and convert x-axis values accordingly.
    if x_axis_label == "samples":
        x_axis_label_pretty = "# Training samples evaluated"
        x_values = [(epoch+1)*n_data for epoch in range(epochs)]
    elif x_axis_label == "epochs":
        x_axis_label_pretty = "# Training epochs evaluated"
        x_values = [epoch + 1 for epoch in range(epochs)]
    else:
        raise Exception(f"Invalid value for x_axis_label: {x_axis_label}")

    # Set x-axis scale.
    if x_axis_scale == "linear":
        x_axis_scale_pretty = "linear"
    elif x_axis_scale == "log":
        x_axis_scale_pretty = "log"
    else:
        raise Exception(f"Invalid x_axis_scale: {x_axis_scale}")

    # Check dataset_name.
    if dataset_name == "mnist":
        dataset_name_pretty = "MNIST"
    elif dataset_name == "frey":
        dataset_name_pretty = "Frey Face"
    else:
        raise Exception(f"Invalid dataset_name: {dataset_name}")

    # Plot training and validation losses.
    plt.plot(x_values, training_losses, c="r")
    plt.plot(x_values, validation_losses, c="b")
    # Set x-axis scale.
    plt.xscale(x_axis_scale_pretty)
    # Set axis labels.
    plt.xlabel(x_axis_label_pretty)
    plt.ylabel(r"$\mathcal{L}$", rotation=90)
    # Set figure title.
    fig_title = dataset_name_pretty + r", $N_{\mathbf{z}}$ = " + str(latent_dim)
    plt.title(fig_title)

    # Set ylim if needed.
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])

    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(os.path.join("images", fig_name))
    plt.show()


def freyface_load_data(split_train=0.8):
    """ Loads the Frey Faces data set.
    Contributed: Jakob Lindén

    Parameters
    ----------
    split_train : float
        Split the data set into training and validation set. split_train is percentage of the training partition.

    Returns
    -------
    x_train : np.ndarray
        Training data set of shape (n_data, height, width)

    y_train : None
        Would be the labels, but the data set has none.

    x_test : np.ndarray
        Validation data set of shape (n_data, height, width)

    y_test : None
        Would be the labels, but the data set has none.

    """
    # Data collected from official website https://cs.nyu.edu/~roweis/data.html
    # as a matlab file
    data_dict = scipy.io.loadmat('data/frey_rawface.mat')
    # data_transposed shape is (image_flattened_dim, n_data).
    data_transposed = data_dict["ff"]
    # data_flattened shape is (n_data, image_flattened_dim).
    data_flattened = np.transpose(data_transposed)
    # From data set documentation.
    height = 28
    width = 20
    # data shape is (n_data, height, width).
    data = data_flattened.reshape(data_flattened.shape[0], height, width)

    # Split the data set into training and validation sets.
    split = np.random.rand(len(data)) < split_train
    x_train, y_train = data[split], None
    x_test, y_test = data[~split], None

    return (x_train, y_train), (x_test, y_test)


def load_dataset(dataset_name):
    """ Wrapper function for loading a data set.
    Contributed: Patrick Jonsson

    Parameters
    ----------
    dataset_name : str
        The name of the data set. mnist for MNIST and frey for Frey Faces

    Returns
    -------
    x_train : np.ndarray
        Training data set of shape (n_data, height, width)

    y_train : np.ndarray or None
        Training labels of shape (n_data,), and None in the case of frey.

    x_test : np.ndarray
        Validation data set of shape (n_data, height, width)

    y_test : np.ndarray or None
        Validation labels of shape (n_data,), and None in the case of frey.

    """
    if dataset_name == "mnist":
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    elif dataset_name == "frey":
        (x_train, y_train), (x_test, y_test) = freyface_load_data(split_train=0.8)
    else:
        raise Exception(f"Invalid dataset_name: {dataset_name}")

    return (x_train, y_train), (x_test, y_test)


def pre_process_dataset(x):
    """ Pre-processes data set - flattening and normalizing.
    Contributed: Jakob Lindén, Márk Csizmadia, Patrick Jonsson

    Parameters
    ----------
    x : np.ndarray
        Training or validation data set of shape (n_data, height, width)

    Returns
    -------
    np.ndarray
        Training or validation data set of shape (n_data, height*width)

    """
    # Flatten and normalize images.
    n_data, height, width = x.shape
    input_dim = height * width
    return x.reshape(-1, input_dim).astype("float32") / 255


def visualize_imgs(x, n_imgs=4):
    """ Pre-processes data set - flattening and normalizing.
    Contributed: Diogo Pinheiro, Jakob Lindén, Márk Csizmadia, Patrick Jonsson

    Parameters
    ----------
    x : np.ndarray
        Training or validation data set of shape (n_data, height, width)

    n_imgs : int
        The number of images to show.

    Returns
    -------
    None
    """
    n_imgs_all = x.shape[0]
    mask = np.random.choice(n_imgs, n_imgs_all)
    x_show = x[mask]

    fig, axes = plt.subplots(n_imgs)
    for idx, ax in enumerate(axes):
        ax.imshow(x_show[idx])
        ax.axis('off')
    plt.show()


def plot_latent_space(vae, img_height, img_width, fig_name, n=30, figsize=15):
    """ Plots latent space. Note that it only works for a 2D latent space.
    Source from https://keras.io/examples/generative/vae/.
    Contributed: Diogo Pinheiro, Jakob Lindén, Márk Csizmadia, Patrick Jonsson

    Parameters
    ----------
    vae : Model
        The trained VAE model.

    n : int
        The number of images to show in rows anc columns (square figure).

    figsize : int
        The size of the pt figure.

    Returns
    -------
    None

    """
    # display a n*n 2D manifold of digits
    scale = 1.0
    figure = np.zeros((img_height * n, img_width * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded, _ = vae.decode(z_sample)
            digit = x_decoded.numpy().reshape(img_height, img_width)
            figure[
                i * img_height : (i + 1) * img_height,
                j * img_width : (j + 1) * img_width,
            ] = digit

    plt.figure(figsize=(figsize, figsize))

    start_range_height = img_height // 2
    end_range_height = n * img_height + start_range_height
    pixel_range_height = np.arange(start_range_height, end_range_height, img_height)
    sample_range_y_height = np.round(grid_y, 1)

    start_range_width = img_width // 2
    end_range_width = n * img_width + start_range_width
    pixel_range_width = np.arange(start_range_width, end_range_width, img_width)
    sample_range_x_width = np.round(grid_x, 1)

    plt.xticks(pixel_range_width, sample_range_x_width)
    plt.yticks(pixel_range_height, sample_range_y_height)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap="Greys_r")
    plt.savefig(os.path.join("images", fig_name))
    plt.show()


def read_config(path_to_config):
    """ Reads the config file in YAML format. Used to just be able to run the code with changing the config file.
    Contributed: Diogo Pinheiro, Jakob Lindén, Patrick Jonsson

    Parameters
    ----------
    path_to_config : str
        The path to the config file.

    Returns
    -------
    config_dict : dict
        A dictionary with the config parameters.
    """
    with open(path_to_config, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as e:
            print(e)

    dataset_name = config["data"]["dataset_name"]
    assert dataset_name in ["frey", "mnist"], f"dataset_name has to be either frey or mnist (got {dataset_name})"
    n_imgs = config["data"]["n_imgs"]

    encoder_hidden_dim = config["vae"]["encoder_hidden_dim"]
    latent_dim = config["vae"]["latent_dim"]
    decoder_hidden_dim = config["vae"]["decoder_hidden_dim"]
    name = config["vae"]["name"]
    assert name in ["VAE_EGDG", "VAE_EGDB"], f"name has to be either VAE_EGDG or VAE_EGDB (got {name})"

    lr = config["train"]["lr"]
    assert not isinstance(lr, str), f"YAML has this bug that it cannot easily read in scientific notation, " \
                                    f"so please input, for instance, 2e-2 as 0.02 (got {lr} of type {type(lr)}" \
                                    f", should be float)"
    epochs = config["train"]["epochs"]
    batch_size = config["train"]["batch_size"]

    neg_elbo_values = config["plot"]["neg_elbo_values"]
    x_axis_label = config["plot"]["x_axis_label"]
    assert x_axis_label in ["epochs", "samples"], \
        f"x_axis_label has to be either epochs or samples (got {x_axis_label})"
    x_axis_scale = config["plot"]["x_axis_scale"]
    assert x_axis_scale in ["linear", "log"], f"x_axis_scale has to be either linear or log (got {x_axis_scale})"
    ylim = config["plot"]["ylim_min"], config["plot"]["ylim_max"]
    assert ylim[0] < ylim[1], f"ylim min has to be smaller than ylim max (got min {ylim[0]} and max {ylim[1]})"
    n_imgs_rec = config["plot"]["n_imgs_rec"]

    bernoulli_decoder_on_mnist = dataset_name == "mnist" and name == "VAE_EGDB"
    gaussian_decoder_on_frey = dataset_name == "frey" and name == "VAE_EGDG"
    assert bernoulli_decoder_on_mnist or gaussian_decoder_on_frey, \
        f"name {name} and dataset_name {dataset_name} do not follow the paper " \
        f"(should be Bernoulli decoder with MNIST, or Gaussian decoder with Frey face"

    config_dict = \
        {"dataset_name": dataset_name,
         "n_imgs": n_imgs,
         "encoder_hidden_dim": encoder_hidden_dim,
         "latent_dim": latent_dim,
         "decoder_hidden_dim": decoder_hidden_dim,
         "name": name,
         "lr": lr,
         "epochs": epochs,
         "batch_size": batch_size,
         "neg_elbo_values": neg_elbo_values,
         "x_axis_label": x_axis_label,
         "x_axis_scale": x_axis_scale,
         "ylim": ylim,
         "n_imgs_rec":n_imgs_rec
         }
    print(f"config_dict: {config_dict}")
    return config_dict


def plot_clusters(vae, x_train, y_test, fig_name, figsize=(12, 10)):
    """ Plots latent space. Note that it only works for a 2D latent space.
    Source from https://keras.io/examples/generative/vae/.
    Contributed: Patrick Jonsson

    Parameters
    ----------
    vae : Model
        The trained VAE model.

    x_train : np.ndarray
        Training data set of shape (n_data, height, width)

    y_test : np.ndarray or None
        Validation labels of shape (n_data,) for mnist

    figsize : tuple
        Changes width & height in inches for plot, preset for now.

    Returns
    -------
    None

    """
    # Vizualises encoded 2D digit clusters
    mu, logvar, z = vae.encode(x_train)
    plt.figure(figsize=figsize)
    plt.scatter(mu[:,0], mu[:,1], c= y_test)
    plt.colorbar()
    plt.xlabel("")
    plt.ylabel("")
    plt.savefig(os.path.join("images", fig_name))
    plt.show()


if __name__ == "__main__":
    # Contributed: Diogo Pinheiro, Jakob Lindén, Márk Csizmadia, Patrick Jonsson
    # Get config.
    path_to_config = "config.yml"
    config_dict = read_config(path_to_config=path_to_config)

    # Saved image identifier.
    img_save_identifier = f"{config_dict['dataset_name']}_z{config_dict['latent_dim']}"

    # Load dataset of choice - either mnist or frey.
    (x_train, y_train), (x_test, y_test) = load_dataset(dataset_name=config_dict["dataset_name"])
    print(f"img shape: height = {x_train.shape[1]}, width = {x_train.shape[2]}")
    print(f"x_train has {x_train.shape[0]} imgs, x_test has {x_test.shape[0]} imgs")

    # Comment out if no visualization needed.
    visualize_imgs(x=x_train, n_imgs=config_dict["n_imgs"])

    # Retrieve the number of images and the image dimensions.
    n_data, height, width = x_train.shape
    input_dim = height * width

    # Pre-process data - flatten and normalize.
    x_train_flattened = pre_process_dataset(x=x_train)
    x_test_flattened = pre_process_dataset(x=x_test)
    print(f"x_train_flattened.shape={x_train_flattened.shape}")
    print(f"x_test_flattened.shape={x_test_flattened.shape}")

    # Create the VAE (depending on config option).
    if config_dict["name"] == "VAE_EGDB":
        vae = VAE_EGDB(input_dim=input_dim,
                       encoder_hidden_dim=config_dict["encoder_hidden_dim"],
                       latent_dim=config_dict["latent_dim"],
                       decoder_hidden_dim=config_dict["decoder_hidden_dim"],
                       name=config_dict["name"])
    else:
        vae = VAE_EGDG(input_dim=input_dim,
                       encoder_hidden_dim=config_dict["encoder_hidden_dim"],
                       latent_dim=config_dict["latent_dim"],
                       decoder_hidden_dim=config_dict["decoder_hidden_dim"],
                       name=config_dict["name"])
    # Plot model if wanted.
    # plot_model(vae, to_file='vae_viz.png', show_shapes=True)

    # Create optimizer.
    # Paper: Stepsizes were adapted with Adagrad [DHS10]; the Adagrad global stepsize parameters were chosen
    # from {0.01,0.02, 0.1} based on performance on the training set in the first few iterations.
    # Use Adam that has built-in weight-decay.
    optimizer = tf.keras.optimizers.Adam(learning_rate=config_dict["lr"])

    # Compile model.
    vae.compile(optimizer)

    # Fit model.
    history = vae.fit(x_train_flattened, x_train_flattened,
                      epochs=config_dict["epochs"], batch_size=config_dict["batch_size"], shuffle=True,
                      validation_data=(x_test_flattened, x_test_flattened))

    # Plot losses.
    plot_lowerbound(
        history=history,
        neg_elbo_values=config_dict["neg_elbo_values"],
        dataset_name=config_dict["dataset_name"],
        latent_dim=config_dict["latent_dim"],
        x_axis_label=config_dict["x_axis_label"],
        x_axis_scale=config_dict["x_axis_scale"],
        n_data=n_data,
        epochs=config_dict["epochs"],
        ylim=config_dict["ylim"],
        fig_name=f"lower_bound_{img_save_identifier}.png")

    # Reconstruct training data.
    x_train_reconstructed_flattened = vae.predict(x_train_flattened)
    x_train_reconstructed = x_train_reconstructed_flattened.reshape(
        n_data, height, width)

    # Visualize the reconstructed images.
    plot_imgs_compare(n_imgs=config_dict["n_imgs_rec"], x=x_train, y=y_train,
                      x_reconstructed=x_train_reconstructed, fig_name=f"reconstruction_{img_save_identifier}.png")

    if config_dict["latent_dim"] == 2:
        # Plot latent space. Only works for latent_dim=2, i.e.: 2D latent space.
        plot_latent_space(vae, img_height=height, img_width=width, fig_name=f"latent_space_{img_save_identifier}.png")
        # Plot clusters, this implementation also only works for latent_dim=2.
        plot_clusters(vae=vae, x_train=x_train_flattened, y_test=y_train,
                      fig_name=f"plot_clusters_{img_save_identifier}.png")