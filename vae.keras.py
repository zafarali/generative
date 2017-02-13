"""
	Variational Autoencoders 

	Following 
	https://github.com/fchollet/keras/blob/master/examples/variational_autoencoder.py
"""
import numpy as np
from keras import layers
from keras.datasets import mnist
from keras import backend as K
from keras.models import Model
from keras.callbacks import TensorBoard, EarlyStopping
from keras.optimizers import Adam
import utils

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
"""
X_in = layers.Input( batch_shape = ( BATCH_SIZE, M ) ) # input layer

if NOISY:
	# only active at training time so we do not worry about
	# trying to not include this during prediction.
	X = layers.noise.GaussianNoise(NOISE_FACTOR)( X_in )
else:
	X = X_in

# gives the most likely z's for a given X Q(z|X)
Q = layers.Dense( output_dim = H_SIZE, activation = 'relu' )( X )


"""
	LATENT SPACE VARIATIONALS
"""
Z_MEAN = layers.Dense( output_dim = L_SIZE ) ( Q ) # mu(X)
Z_LOG_VAR = layers.Dense( output_dim = L_SIZE ) ( Q ) # sigma(X)


# sampling from the latent space
def sampling(args):
	z_mean, z_var = args 

	# sample from N(0, 1)
	eps = K.random_normal( \
		shape = ( BATCH_SIZE, L_SIZE ), \
		mean = 0, \
		std = EPS )

	# use the reparametrization trick:
	return z_mean + K.exp(Z_LOG_VAR /2 ) * eps

# layer that samples from the latent space
# given the previous mean and sigma functions
Z = layers.Lambda( sampling, output_shape= ( L_SIZE, ) ) ( [ Z_MEAN , Z_LOG_VAR ] )

"""
	DECODER LAYERS
	using _ to denote that these are layers, 
	keeping this separate will allow us to use this in the future
	for sampling from P(X|z) to get P(X)
"""

_P = layers.Dense( output_dim = H_SIZE, activation = 'relu')
_f = layers.Dense( output_dim = M , activation = 'sigmoid' )

P = _P( Z ) # take in the latent information - P(X|z)
X_generated = _f( P ) # obtain a function that gives ~x = f(X)




def ELBO(x_true, x_pred):
	# since the feature space is not binary we calculate the pixelwise
	# distance between the two as the error
	# reconstruction_loss = K.sum(K.binary_crossentropy(x_true, x_pred),axis=1)
	reconstruction_loss = K.sum(K.square(x_true - x_pred), axis=-1)
	
	# Kullback-Leibler divergence between a normal with mean mu and sigma
	# from a normal with mean 0 and sigma 1
	# -0.5 ( Tr(Sigma) + (mu)^T(mu) - 1 - log (det(sigma)) )
	KL_loss = - 0.5 * K.sum(1 +  Z_LOG_VAR -  K.square(Z_MEAN) - K.exp(Z_LOG_VAR), axis=-1)
	
	return reconstruction_loss + KL_loss

callbacks = [TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False),
			EarlyStopping(monitor='val_loss', patience=1, verbose=1, mode='auto')]



# flattening the dataset
# these will act as the inputs
X_train = (X_train.astype('float32') / 255).reshape(-1, M)
X_test = (X_test.astype('float32') / 255).reshape(-1, M)


adam = Adam(lr=0.005)

VAE = Model(X_in, X_generated)
VAE.compile(optimizer=adam, loss=ELBO)

try:
	VAE.fit(X_train, X_train, batch_size=BATCH_SIZE, shuffle=True, \
		nb_epoch = EPOCHS, validation_data = ( X_test, X_test ) )
except KeyboardInterrupt as e:
	print('Training Stopped Early')

# encode the images into a latent space mean.
ENCODER = Model(X_in, Z_MEAN)

# decode latent space information into images
decoder_in = layers.Input(shape=(L_SIZE,))
P = _P(decoder_in) # 
X_generated = _f(P) # ~x = f(z)

GENERATOR = Model(decoder_in, X_generated)

f, ax = utils.data_on_latent_space(ENCODER, X_test, Y_test, batch_size=BATCH_SIZE)
# f.colorbar()

f.savefig('latent_viz.pdf')

f, ax = utils.manifold_2D(GENERATOR)
f.savefig('manifold.pdf')




