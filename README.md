# DD2434-VAE-Project (Re-Autoencoding Variational Bayes)

Small Project in DD2434 Advanced Machine Learning by Diogo Pinheiro, Jakob Lindén, Márk Csizmadia, and Patrick Jonsson.

## Installation

In the CLI of your choice

```bash
git clone https://github.com/jakobGTO/DD2434-VAE-Project.git
```

Create a virtual environment (or in your base environment) with Anaconda or any virtual environment manager (see Anaconda example below): 

```bash
conda create -n dd2434_vae_project python=3.8
source activate dd2434_vae_project
pip install -r requirements.txt
```

## Usage

The usage changed, there is only one script now - VAE.py. The code from the main branch has been reorganized into this one script. The code is the same, but a little more cleaned up and with more comments.

```bash
cd DD2434-VAE-Project
python VAE.py
```

Try to train for 50 epoch with a batch size of 32 with the same optimizer as it is now. I start out with around 1.2928 loss, on the 2 epoch it's around 0.2682, and by the 50th epoch it's around 0.1417.

## Some other section

bla bla
