"""
	Conditional Variational Autoencoder
"""
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras import backend as K
from keras.models import Model
from keras import layers
from keras.objectives import binary_crossentropy
from keras.utils import np_utils
from keras.optimizers import Adam
import utils

# obtain the data
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

M = X_train[0].shape[0] * X_train[0].shape[1] # feature size

# flattening the dataset
# these will act as the inputs
X_train = (X_train.astype('float32') / 255).reshape(-1, M)
X_test = (X_test.astype('float32') / 255).reshape(-1, M)

# conver labels to categorical information
Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)

OUT_SIZE = Y_train.shape[1]
BATCH_SIZE = 100 # batch size to use during training
H_SIZE = 256 # number of nodes in the hidden layer
L_SIZE = 2 # size of the latent layer
EPS = 1 # standard deviation of the sampler
EPOCHS = 50 # number of training steps
NUMERICAL_STABILIZATION = 1e-6
NOISY = False # we want to add noise to our images before feeding them in.
NOISE_FACTOR = 0.1 # the factor of noise to add.

"""
	ENCODER LAYERS
	Q(z | X, Y)
	# here goal is to generate Y, given X
	# therefore now Y is the image and X is the label.
"""

X = layers.Input( batch_shape=( BATCH_SIZE , M ) )
Y = layers.Input( batch_shape=( BATCH_SIZE , OUT_SIZE ) )
in_merged = layers.merge( [ X , Y ] , mode='concat', concat_axis=1 )

Q = layers.Dense( H_SIZE , activation='relu' )( in_merged )

"""
	LATENT SPACE VARIATIONALS
	Mu( X, Y ) , Sigma( X, Y )
"""

Z_MEAN = layers.Dense( L_SIZE )( Q )
Z_LOG_VAR = layers.Dense( L_SIZE )( Q )


# sampling from the latent space
# including reparameterization trick

def sample(args):
	z_mu, z_log_sigma = args

	# sample from N(0, 1)
	eps = K.random_normal( \
		shape = ( BATCH_SIZE, L_SIZE ), \
		mean = 0, \
		std = EPS )
	return z_mu + K.exp( z_log_sigma / 2.0 ) * eps


Z = layers.Lambda( sample )( [ Z_MEAN , Z_LOG_VAR ] )
Z_Y = layers.merge( [ Z , Y ] , mode='concat', concat_axis=1)

"""
	DECODER LAYERS
	P( Y | Z , X )
"""

P_ = layers.Dense( H_SIZE , activation='relu' )
f_ = layers.Dense( M , activation='sigmoid' )

P = P_( Z_Y )
X_generated = f_( P )

"""
	ELBO LOSS
"""

def ELBO(true, pred):
	# since the feature space is not binary we calculate the pixelwise
	# distance between the two as the error
	# reconstruction_loss = K.sum(K.binary_crossentropy(true, pred),axis=1)
	reconstruction_loss = K.sum(K.square(true - pred), axis=-1)
	
	# Kullback-Leibler divergence between a normal with mean mu and sigma
	# from a normal with mean 0 and sigma 1
	# -0.5 ( Tr(Sigma) + (mu)^T(mu) - 1 - log (det(sigma)) )
	KL_loss = - 0.5 * K.sum(1 +  Z_LOG_VAR -  K.square(Z_MEAN) - K.exp(Z_LOG_VAR), axis=-1)
	
	return reconstruction_loss + KL_loss

# ADAM OPTIMIZER

adam = Adam(lr=0.005)

# TRAIN VAE
# image first, labels second
# note that X = layer representing conditions
# and Y = layer representing images
# wheras X_train = images and Y_train labels.
CVAE = Model( [ X, Y ], X_generated )
CVAE.compile(optimizer=adam, loss=ELBO)

print(CVAE.summary())

try:
	# FIT
	CVAE.fit( [ X_train , Y_train ], X_train, \
		batch_size=BATCH_SIZE, nb_epoch=EPOCHS, validation_data = ( [ X_test, Y_test ], X_test ) )
except KeyboardInterrupt as e:
	print('Training Stopped Early')

# use encoder encode the images into a latent space mean.

ENCODER = Model( [ X, Y ], Z_MEAN )

# use decoder to decode latent space information into images

z_in = layers.Input( shape=( L_SIZE , ) )
y_in = layers.Input( shape=( OUT_SIZE , ) )
decoder_in = layers.merge( [ z_in , y_in ] , mode='concat', concat_axis=1 )
decoder_hidden = P_( decoder_in )
decoder_out = f_( decoder_hidden )
DECODER = Model( [ z_in, y_in ] , decoder_out )


"""
	PLOTTING
"""
# ensure that encoder and decoder have a .predict() function on each.
x_encoded = ENCODER.predict( [ X_train, Y_train ], batch_size=BATCH_SIZE )
f, ax = utils.data_on_latent_space( x_encoded, np_utils.probas_to_classes(Y_train) )
# f.colorbar()

f.savefig('latent_viz.cvae.pdf')

n_digits = 10
n = 5
latent_range=(0.05, 0.95)
digit_shape=(28,28)


figure = np.zeros((digit_shape[0] * n_digits, digit_shape[1] * n))

for i in range(1, n_digits+1):
	digit_code = np.zeros(n_digits) # encode the digit we are trying to generate
	digit_code[i-1] = 1
	digit_code = digit_code.reshape(1, -1)
	for k in range(n):
		z_noise = np.random.randn(L_SIZE)
		z_noise = z_noise.reshape(1, -1)
		x_generated = DECODER.predict( [ z_noise , digit_code ] )
		digit = x_generated.reshape(digit_shape)
		figure[ ( i - 1 ) * digit_shape[0] : i * digit_shape[0] , \
				k * digit_shape[1] : ( k + 1 ) * digit_shape[1] ] = digit


f = plt.figure(figsize=(10, 10))
ax = f.add_subplot(111)
ax.imshow(figure, cmap='Greys_r')
# f, ax = utils.manifold_2D(GENERATOR)
f.savefig('manifold.cvae.pdf')
f.show()




