# DD2434-VAE-Project

## Gameplan

### 1) Normal multilayered perceptron as encoder (equation 12)
#### Takes input data, calculate parameters of normaldistribution (mu,sigma)
#### for the latent variable Z (z = mu + sigma * epsilon ~ N(0,1)) 

### 2) Normal multilayered perceptron as decoder (equation 12)
#### Sample from normaldistribution with previously calculated mu,sigma 

### 3) Train network

### 4) Use model and visualize results