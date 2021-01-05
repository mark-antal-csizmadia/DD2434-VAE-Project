from keras.layers import Dense, Lambda
from keras.models import Model, load_model
from keras import backend as kb
from keras.utils import plot_model
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import datetime
import scipy.io

# Global constants.
# Epsilon: avoid numerical instability issues such as log(0) -> nan.
EPSILON = 1e-6
tf.random.set_seed(221)


class VAE(Model):
    """ Variational Auto-Encoder class via subclassing keras.Model.
    """

    def __init__(self, input_dim, encoder_hidden_dim, latent_dim, decoder_hidden_dim, name):
        """ Init function of the VAE class.

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
        super(VAE, self).__init__(name=name)
        # Xavier initialization (https://www.deeplearning.ai/ai-notes/initialization/, see towards the end of the
        # article)
        self.initializer = tf.initializers.VarianceScaling(scale=2.0)
        # Create the VAE layers.
        self.encoder_hidden = Dense(
            units=encoder_hidden_dim, activation="relu", kernel_initializer=self.initializer)
        self.mu = Dense(
            latent_dim, kernel_initializer=self.initializer, name="mu")
        self.sigma = Dense(
            latent_dim, kernel_initializer=self.initializer, name="sigma")
        self.z = Lambda(self.get_latent, output_shape=(latent_dim,), name="z")
        self.decoder_hidden = Dense(
            decoder_hidden_dim, activation='relu', kernel_initializer=self.initializer)
        # The decoder is a Bernoulli MLP. Has binary cross-entropy loss, see later.
        self.reconstruction = \
            Dense(input_dim, activation='sigmoid',
                  kernel_initializer=self.initializer, name="reconstruction")

    def get_latent(self, gauss_params):
        """ Function to re-parametrize mu,sigma by sampling from Gaussian

        Parameters
        -----------
        gauss_params : tf.Tensor
           mu and sigma of q(z|x), both have shape (batch_size, latent_dim) where batch_size is the
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
        mu, sigma = gauss_params
        # Sample epsilon and calculate latent vector Z (equation 10)
        eps = kb.random_normal(shape=(kb.shape(mu)[0], kb.shape(mu)[1]))
        # standard deviation = sqrt(sigma) ; using the exponential assures that the result is positive
        std = kb.exp(0.5*sigma)
        z = mu + std * eps
        return z

    def call(self, inputs):
        """ Overrides the call function of keras.Model via subclassing. Gets called at each optimization step.
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
        # encoder_hidden shape=(batch_size, encoder_hidden_dim)
        encoder_hidden = self.encoder_hidden(inputs)
        # mu shape=(batch_size, latent_dim)
        mu = self.mu(encoder_hidden)
        # sigma shape=(batch_size, latent_dim)
        sigma = self.sigma(encoder_hidden)
        # z shape=(batch_size, latent_dim)
        z = self.z([mu, sigma])
        # decoder_hidden shape=(batch_size, decoder_hidden_dim)
        decoder_hidden = self.decoder_hidden(z)
        # decoder_hidden shape=(batch_size, input_dim)
        reconstruction = self.reconstruction(decoder_hidden)

        # Create the loss tensors.
        # Computes negative! KL Divergence Loss (First part of equation 24, but negative here to have -KLD)
        negative_kl_loss = kb.mean(
            0.5 * kb.sum(-sigma + kb.exp(sigma) + kb.square(mu) - 1, axis=1))
        # Computes the Binary Cross-Entropy Loss that is equivalent to the negative binary loglikelihood
        # (comes from the sigmoid activation function of the Bernoulli MLP layer in the decoder)
        binary_cross_entropy_loss = kb.mean(-kb.sum(inputs * kb.log(
            reconstruction + EPSILON) + (1 - inputs) * kb.log(1 - reconstruction + EPSILON), axis=1))
        # Create the overall VAE loss.
        # ELBO in the paper is ELBO = KLD + Log_likelihood, ELBO in the paper is maximized
        # We minimize the negative ELBO: - ELBO = - KLD - Log_likelihood = - KLD - Log_likelihood = - KLD + BCE
        vae_loss = binary_cross_entropy_loss + negative_kl_loss

        self.add_loss(vae_loss)

        # Return the reconstructed observations.
        return reconstruction


def plot_imgs_compare(n_imgs, x, y, x_reconstructed, save_img):
    """ Overrides the call function of keras.Model via subclassing. Gets called at each optimization step.
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
    y_show = y[idx_imgs]
    x_reconstructed_show = x_reconstructed[idx_imgs]

    # Create figure.
    fig_height = n_imgs*5
    fig_width = 10
    fig, axs = plt.subplots(n_imgs, 2, figsize=(fig_height, fig_width))
    fig.tight_layout()

    # Plot the subplots of the figure.
    for n_idx, _ in enumerate(idx_imgs):
        ax_left = axs[n_idx, 0]
        ax_right = axs[n_idx, 1]
        ax_left.imshow(x_reconstructed_show[n_idx])
        ax_right.imshow(x_show[n_idx])
        ax_left.set_title(f"Reconstructed")
        ax_right.set_title(f"Label: {y_show[n_idx]}")
        ax_left.axis('off')
        ax_right.axis('off')

    # Save image if desired.
    if save_img:
        plt.savefig("images/reconstruction.png")

    plt.show()


def plot_lowerbound(history, neg_elbo_values, dataset_name, latent_dim, x_axis_label, x_axis_scale, n_data, epochs,
                    ylim):
    """ Plot likelihood lower bound. The loss curves of training and validation.

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

    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])

    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig("images/lower_bound.png")
    plt.show()


def freyface():
    # Data collected from official website https://cs.nyu.edu/~roweis/data.html
    # as a matlab file
    data = scipy.io.loadmat('data/frey_rawface.mat')
    height = 20
    width = 28

    input_dim = height * width
    data = data["ff"].T.reshape((-1, input_dim))
    data = data.astype('float32')/255

    split = np.random.rand(len(data)) < 0.9
    train = data[split]
    test = data[~split]

    return train,test,height,width


if __name__ == "__main__":
    # Uncomment for freyface
    
    """x_train_flattened,x_test_flattened,ff_height,ff_width = freyface()
    n_data, height, width = len(x_train_flattened),ff_height,ff_width
    input_dim = ff_height * ff_width"""
    
    # Uncomment for mnist
    # Import the data set.
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    n_data, height, width = x_train.shape

    input_dim = height * width
    # Reshape data to (60000,784) in mnist case
    x_train_flattened = x_train.reshape(-1, input_dim).astype("float32") / 255
    x_test_flattened = x_test.reshape(-1, input_dim).astype("float32") / 255

    #x_train = x_train[np.random.randint(x_train.shape[0],size=1000)]


    # Create the VAE.
    encoder_hidden_dim = 100
    latent_dim = 20
    decoder_hidden_dim = 100
    vae = VAE(input_dim=input_dim, encoder_hidden_dim=encoder_hidden_dim, latent_dim=latent_dim,
              decoder_hidden_dim=decoder_hidden_dim, name="vae")
    # Plot model if wanted. Something is not okay with this now, leave it commented.
    # plot_model(vae, to_file='vae_viz.png', show_shapes=True)

    # Create optimizer.
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    # Compile model.
    vae.compile(optimizer)

    # Fit model.
    epochs = 200
    batch_size = 32
    history = vae.fit(x_train_flattened, x_train_flattened,
                      epochs=epochs, batch_size=batch_size, shuffle=True, validation_data=(x_test_flattened, x_test_flattened))

    neg_elbo_values = True
    dataset_name = "mnist"
    x_axis_label = "samples"
    x_axis_scale = "log"
    ylim = (-150, -100)

    plot_lowerbound(
        history=history,
        neg_elbo_values=neg_elbo_values,
        dataset_name=dataset_name,
        latent_dim=latent_dim,
        x_axis_label=x_axis_label,
        x_axis_scale=x_axis_scale,
        n_data=n_data,
        epochs=epochs,
        ylim=ylim)

    # Reconstruct training data.
    x_train_reconstructed_flattened = vae.predict(x_train_flattened)
    x_train_reconstructed = x_train_reconstructed_flattened.reshape(
        n_data, height, width)

    # Visualize the reconstructed images.
    plot_imgs_compare(n_imgs=10, x=x_train, y=y_train,
                      x_reconstructed=x_train_reconstructed, save_img=True)


    """
    This part does not work yet.
    # This part shows how to save a model and then use it for inference again without further training. A saved model
    # can be further trained as well.
    st = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    model_name = "vae"
    vae.save(model_name)

    # It can be used to reconstruct the model identically.
    #vae_reconstructed = load_model(model_name)
    # It's tricky to save and load models with Lambda layers.
    vae_reconstructed = load_model(model_name, custom_objects={'get_latent': VAE.get_latent})

    # Let's check:
    test_input = x_train_flattened[:10]
    test_target = y_train[:10]
    np.testing.assert_allclose(vae.predict(test_input), vae_reconstructed.predict(test_input))

    # The reconstructed model is already compiled and has retained the optimizer
    # state, so training can resume:
    vae_reconstructed.fit(test_input, test_target)
    test_input_reconstructed = vae_reconstructed.predict(test_input)"""
