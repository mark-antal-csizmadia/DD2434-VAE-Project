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

Use the main script to train a Variational Auto-Encoder (VAE). Set the encoder and decoder network parameters, the training parameters (e.g.: learning rate) the type of decoder (Bernoulli or Gaussian) and the data set (MNIST or Frey face) in the code.

```bash
cd DD2434-VAE-Project
python main.py
```


## Description

The code is a replication of the research paper titled Auto-Encoding Variational Bayes (https://arxiv.org/abs/1312.6114). See the reproduced results in the report (some are also located at  images/).
