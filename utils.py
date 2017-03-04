import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def data_on_latent_space(encoded, categories, ax=None):
	"""
		plots the data in the latent space
		encoded: first two dimensions of the data encoded
		categories: the categories for each datapoint to (for visualization purposes)
		batch_size[=32]: the batch size for the predictions
		ax[=None]: axis to add the plot to
	"""

	if not ax:
		f = plt.figure(figsize=(6, 6))
		ax = f.add_subplot(111)
	else:
		f = None

	ax.scatter(encoded[:,0], encoded[:,1], c=categories)
	
	return f, ax

def manifold_2D(generator, ax=None, n=15, shape=(28,28), latent_space='gaussian', latent_range=(0.05, 0.95)):
	""" display a 2D manifold of the digits
		@params:
			generator: a generator with a .predict() function
			ax[=None]: axis to add the plot to
			n[=15]: number of samples to generate for each dimension
			shape[=(28,28)]: reshape of the sample
			latent_space[='gaussian']
			latent_range[=(0.05,0.95)]

		@returns:
			matplotlib axes with the figure added.
	"""
	digit_size = shape[0]
	figure = np.zeros((digit_size * n, digit_size * n))
	# linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
	# to produce values of the latent variables z, since the prior of the latent space is Gaussian
	if latent_space == 'gaussian':
		grid_x = stats.norm.ppf(np.linspace(latent_range[0], latent_range[1], n))
		grid_y = stats.norm.ppf(np.linspace(latent_range[0], latent_range[1], n))
	else:
		raise NotImplementedError('Unknown Latent Space not yet implemented')

	for i, yi in enumerate(grid_x):
		for j, xi in enumerate(grid_y):
			z_sample = np.array([[xi, yi]])
			x_decoded = generator.predict(z_sample)
			digit = x_decoded[0].reshape((digit_size, digit_size))
			figure[i * digit_size: (i + 1) * digit_size,
				   j * digit_size: (j + 1) * digit_size] = digit

	if not ax:
		f = plt.figure(figsize=(10, 10))
		ax = f.add_subplot(111)
	else:
		f = None

	ax.imshow(figure, cmap='Greys_r')

	return f, ax
