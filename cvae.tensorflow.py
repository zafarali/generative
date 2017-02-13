"""
	Variational Autoencoders 

	Following 
	https://github.com/fchollet/keras/blob/master/examples/variational_autoencoder.py
"""
import numpy as np
from keras.dataset import mnist
import tensorflow as tf
import utils

slim = tf.contrib.slim

# obtain the data
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()


M = X_train[0].shape[0] * X_train[0].shape[1] # feature size
BATCH_SIZE = 100 # batch size to use during training
H_SIZE = 256 # number of nodes in the hidden layer
L_SIZE = 2 # size of the latent layer
EPS = 1 # standard deviation of the sampler
EPOCHS = 50 # number of training steps
NUMERICAL_STABILIZATION = 1e-6
NOISY = True # we want to add noise to our images before feeding them in.
NOISE_FACTOR = 0.1 # the factor of noise to add.

"""
	ENCODER LAYERS
	Q(z | X, Y)
"""



"""
	LATENT SPACE VARIATIONALS
	Mu( X, Y ) , Sigma( X, Y )
"""



# sampling from the latent space
# including reparameterization trick



"""
	DECODER LAYERS
	P( Y | Z , X )
"""


"""
	ELBO LOSS
"""



# flattening the dataset
# these will act as the inputs
X_train = (X_train.astype('float32') / 255).reshape(-1, M)
X_test = (X_test.astype('float32') / 255).reshape(-1, M)


# ADAM OPTIMIZER


# TRAIN VAE

try:
	# FIT
except KeyboardInterrupt as e:
	print('Training Stopped Early')

# use encoder encode the images into a latent space mean.


# use decoder to decode latent space information into images


"""
	PLOTTING
"""
# ensure that encoder and decoder have a .predict() function on each.
# f, ax = utils.data_on_latent_space(ENCODER, X_test, Y_test, batch_size=BATCH_SIZE)
# f.colorbar()

# f.savefig('latent_viz.pdf')

# f, ax = utils.manifold_2D(GENERATOR)
# f.savefig('manifold.pdf')




