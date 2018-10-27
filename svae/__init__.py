import os

import numpy as np
import tensorflow as tf

class SVAE:
	
	def __init__(
		self, checkpoint_dir, log_dir, img_shape=(128, 128, 3), num_latents=32, 
		num_classes=2):
		
		self.checkpoint_dir = checkpoint_dir
		self.img_shape = img_shape
		self.num_latents = num_latents
		self.num_classes = num_classes
		
		with tf.variable_scope('svae', reuse=tf.AUTO_REUSE):
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
			self._create_network()
			self._create_losses()
			self._create_optimizers()
		
		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())
		
		self._load_checkpoint()
        
		self.summary_writer = tf.summary.FileWriter(log_dir, self.sess.graph)
		
	def _load_checkpoint(self):
		"""
		Checks if a model checkpoint exists, and if so, alters 'self.sess' to
		reflect the appropriate state of the Tensorflow computation graph.
		"""
		self.saver = tf.train.Saver(max_to_keep=5)
		checkpoint = tf.train.get_checkpoint_state(self.checkpoint_dir)

		# If checkpoint exists and is reachable, load checkpoint state into 'sess'
		if checkpoint and checkpoint.model_checkpoint_path:
				self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
				print('loaded checkpoint: {}'.format(checkpoint.model_checkpoint_path))
		else:
				print(
						'Could not find old checkpoint. '
						'Creating new checkpoint directory.'
				)
				if not os.path.exists(self.checkpoint_dir):
						os.mkdir(self.checkpoint_dir)
        
        def _save_checkpoint(self):
		self.saver.save(
			self.sess,
			self.checkpoint_dir,
			global_step=self.global_step)
						
	def _create_encoder_network(self, x, reuse=tf.AUTO_REUSE):
		"""
		Create computation graph that returns output tensors of the recognition
		network: a tensor of means and a tensor of log standard deviations that
		define the factorized latent distribution q(z).
		"""
		with tf.variable_scope('encoder', reuse=reuse):
			# If input images were reshaped into vectors, reshape them back into
			# their original dimensions for convolutional layers.
			# height, width, channels = self.img_shape
			# x_reshaped = tf.reshape(x, [-1, height, width, channels])

			# Ex: filters=32, kernel_size=4, stride=2
			x = tf.layers.Conv2D(32, 4, 2, 'same', activation=tf.nn.relu,)(x)
			x = tf.layers.BatchNormalization(axis=-1, momentum=0.1, epsilon=1e-5)(x)
			x = tf.layers.Conv2D(64, 4, 2, 'same', activation=tf.nn.relu,)(x)
			x = tf.layers.BatchNormalization(axis=-1, momentum=0.1, epsilon=1e-5)(x)
			x = tf.layers.Conv2D(128, 4, 2, 'same', activation=tf.nn.relu,)(x)
			x = tf.layers.BatchNormalization(axis=-1, momentum=0.1, epsilon=1e-5)(x)
			x = tf.layers.Conv2D(128, 4, 2, 'same', activation=tf.nn.relu,)(x)
			x = tf.layers.BatchNormalization(axis=-1, momentum=0.1, epsilon=1e-5)(x)
			x = tf.layers.Conv2D(256, 4, 2, 'same', activation=tf.nn.relu,)(x)
			x = tf.layers.BatchNormalization(axis=-1, momentum=0.1, epsilon=1e-5)(x)
			x = tf.layers.Conv2D(512, 4, activation=tf.nn.relu,)(x)
			x = tf.layers.BatchNormalization(axis=-1, momentum=0.1, epsilon=1e-5)(x)

			# Final convolutions downsize each channels' dimensions to a 1x1 patch,
			# resulting in a final tensors with shape (batch_size, 1, 1, num_latents)
			z_mean = tf.layers.Conv2D(self.num_latents, 1)(x) 

			# log_sigma used for numerical stability
			z_log_sigma = tf.layers.Conv2D(self.num_latents, 1)(x)

			# Reshape outputs to remove extra dimensions. This is necessary
			# for the _sample_z function.
			z_mean = tf.reshape(
				z_mean, shape=(-1, self.num_latents), name='z_mean')
			z_log_sigma = tf.reshape(
				z_log_sigma, shape=(-1, self.num_latents), name='z_log_sigma')

			return (z_mean, z_log_sigma)
		
	def _create_decoder_network(self, z, reuse=tf.AUTO_REUSE):
		"""
		Create computation graph that accepts tensors for sampled latent vectors
		and returns output logit tensors. Activations for these output logits are
		applied separately when creating the full autoencoder network and loss
		corresponding loss function.
		"""
		with tf.variable_scope('decoder', reuse=reuse):
			# A mini-batch of sampled latent vectors has shape
			# (batch_size, num_latents) and need to be reshaped into num_latents
			# 1x1 channels
			z = tf.reshape(z, shape=(-1, 1, 1, self.num_latents))

			# Ex: filters=32, kernel_size=1, stride=1
			z = tf.layers.Conv2DTranspose(
				512, 1, 1, 'valid', activation=tf.nn.relu,)(z)
			z = tf.layers.BatchNormalization(
				axis=-1, momentum=0.1, epsilon=1e-5)(z)
			z = tf.layers.Conv2DTranspose(
				256, 4, 1, 'valid', activation=tf.nn.relu,)(z)
			z = tf.layers.BatchNormalization(
				axis=-1, momentum=0.1, epsilon=1e-5)(z)
			z = tf.layers.Conv2DTranspose(
				128, 4, 2, 'same', activation=tf.nn.relu,)(z)
			z = tf.layers.BatchNormalization(
				axis=-1, momentum=0.1, epsilon=1e-5)(z)
			z = tf.layers.Conv2DTranspose(
				128, 4, 2, 'same', activation=tf.nn.relu,)(z)
			z = tf.layers.BatchNormalization(
				axis=-1, momentum=0.1, epsilon=1e-5)(z)
			z = tf.layers.Conv2DTranspose(
				64, 4, 2, 'same', activation=tf.nn.relu,)(z)
			z = tf.layers.BatchNormalization(
				axis=-1, momentum=0.1, epsilon=1e-5)(z)
			z = tf.layers.Conv2DTranspose(
				32, 4, 2, 'same', activation=tf.nn.relu,)(z)
			x = tf.layers.Conv2DTranspose(
				3, 4, 2, 'same', activation=None,)(z)

			# Flatten the deconvolved tensors into vectors
#       height, width, channels = self.img_shape
#       x_out_logit = tf.reshape(
#         x, shape=(-1, height*width*channels), name='x_out_logit')

			x_out_logit = tf.identity(x, name='x_out_logit')

			return x_out_logit
		
	def _sample_z(self, z_mean, z_log_sigma):
		"""
		Reparametrization trick to sample from latent distribution.
		sample = mean + standard_deviation * gaussian_noise 
		where gaussian_noise ~ N(0, 1)
		"""
		eps_shape = tf.shape(z_mean)
		eps = tf.random_normal(eps_shape, 0, 1, dtype=tf.float32 )

		# z = mu + sigma * epsilon
		z = tf.add(z_mean,
							 tf.multiply(tf.exp(z_log_sigma), eps), name='z_sample')
		return z

	def _create_network(self):
		"""
		Define the entire computation graph that includes the recognition network,
		latent distribution sampling, and generator network.
		"""
		height, width, channels = self.img_shape
		self.x_input = tf.placeholder(
			tf.float32, shape=[None, height, width, channels], name='x_input')
		self.y_input = tf.placeholder(
			tf.float32, shape=[None, 2], name='y_input')
		self.learning_rate = tf.placeholder(
			tf.float32, name='learning_rate')
		self.beta = tf.placeholder(
			tf.float32, name='beta')

		# Get tensors for latent distribution parameters
		self.z_mean, self.z_log_sigma = self._create_encoder_network(self.x_input)

		# Sample latent vectors from their corresponding distributions
		self.z = self._sample_z(self.z_mean, self.z_log_sigma)
		
		self.y_logits = self._create_classifier_network(self.z)
		
		# When not training, generate deterministic predictions by
		# substituting the sampled latent vectors with the means of the
		# distribution.
		y_logits_denoised = self._create_classifier_network(self.z_mean)
		self.y_pred_denoised = tf.nn.softmax(y_logits_denoised)

		# Using latent sample, create reconstructed pixel values. These noisy
		# reconstructions are only used to calculate loss when training the model.
		self.x_out_logit = self._create_decoder_network(self.z)
		self.x_out = tf.nn.sigmoid(self.x_out_logit, name='x_out')

		# When not training, generate deterministic reconstructions by
		# substituting the sampled latent vectors with the means of the
		# distribution.
		self.x_out_logit_denoised = self._create_decoder_network(
			self.z_mean, reuse=True)
		self.x_out_denoised = tf.nn.sigmoid(
			self.x_out_logit_denoised, name='x_out_denoised')
			
	def _create_classifier_network(self, z):
		with tf.variable_scope('classifier', reuse=tf.AUTO_REUSE):
			y_logits = tf.layers.Dense(self.num_classes, name='y_logits')(z)
		return y_logits
	
	def _create_losses(self):
        
        summary_ops = list() 
		
		# Flatten each input image into a vector
		height, width, channels = self.img_shape
		flat_shape = [-1, height*width*channels]
		x_input_flattened = tf.reshape(self.x_input, flat_shape)
		x_out_logit_flattened = tf.reshape(self.x_out_logit, flat_shape)
		
		# Reconstruction loss
		reconstruction_loss = tf.nn.sigmoid_cross_entropy_with_logits(
			labels=x_input_flattened, logits=x_out_logit_flattened)
		reconstruction_loss = tf.reduce_sum(reconstruction_loss, 1)
		self.reconstruction_loss = tf.reduce_mean(
			reconstruction_loss, name='reconstruction_loss')
        
        summary_ops.append(tf.summary.scalar('reconstruction loss', self.reconstruction_loss))
		
		# Binary cross-entropy loss
		cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
		 labels=self.y_input, logits=self.y_logits)
		self.cross_entropy_loss = tf.reduce_mean(
			cross_entropy_loss, name='cross_entropy_loss')
        
        summary_ops.append(tf.summary.scalar('cross_entropy loss', self.cross_entropy))

		# Latent loss
		z_log_sigma_sq = tf.square(self.z_log_sigma)
		latent_loss = -0.5 * tf.reduce_sum(1 + z_log_sigma_sq
																			 - tf.square(self.z_mean)
																			 - tf.exp(z_log_sigma_sq), 1)
		self.latent_loss = tf.reduce_mean(latent_loss)
        
        summary_ops.append(tf.summary.scalar('latest_loss', self.latent_loss))
		
		self.classification_loss = \
			self.cross_entropy_loss + self.beta * self.latent_loss
        
        summary_ops.append(tf.summary.scalar('classification_loss', self.classification_loss))
        
        self.summary_ops_merged = tf.summary.merge(summary_ops)
			
	def _create_optimizers(self):
		
		optimizer = tf.train.AdamOptimizer
		classifier_optimizer = optimizer(self.learning_rate)
		self.classifier_optimizer = classifier_optimizer.minimize(
			self.classification_loss)
		
		decoder_weights = tf.get_collection(
			tf.GraphKeys.TRAINABLE_VARIABLES, 'svae/decoder')
		decoder_optimizer = optimizer(self.learning_rate)
		self.decoder_optimizer = decoder_optimizer.minimize(
			self.reconstruction_loss, var_list=decoder_weights, global_step=self.global_step)
		
	def _partial_fit_classifier(self, x_batch, y_batch, learning_rate, beta):
		"""
		Train encoder and classifier networks based on minibatch
		of training data.

		Parameters
		----------
		x_batch : array-like, shape = [batch_size, height, width, channels]
			A minibatch of input images.
		y_batch : array-like, shape = [batch_size, num_classes]
			A one-hot encoded matrix of training labels.
		"""
		
		feed_dict = {
			self.x_input: x_batch,
			self.y_input: y_batch,
			self.learning_rate: learning_rate,
			self.beta: beta
		}
		_ = self.sess.run(self.classifier_optimizer, feed_dict=feed_dict)
        
        step = self.sess.run(self.global_step)
		summary_str = self.sess.run(self.summary_ops_merged, feed_dict=feed_dict)
		self.summary_writer.add_summary(summary_str, step)

	def fit_classifier(
		self, x, y, num_epochs=5, batch_size=256,
		learning_rate=1e-3, beta=1):
		"""
		Train encoder and classifier networks.

		Parameters
		----------
		x : array-like, shape = [num_samples, height, width, channels]
			A set of input images.
		y : array-like, shape = [num_samples, num_classes]
			A one-hot encoded matrix of training labels.
		"""
		# Shuffle x and y
		num_samples = len(x)
		for epoch in range(num_epochs):
			random_indices = np.random.permutation(num_samples)
			x = x[random_indices]
			y = y[random_indices]

			# Split x and y into batches
			num_batches = num_samples // batch_size
			indices = [[k, k+batch_size] for k in range(0, num_samples, batch_size)]
			indices[-1][-1] = num_samples
			x_batches = [x[start:end] for start, end in indices]
			y_batches = [y[start:end] for start, end in indices]
			
			print(f'Training epoch {epoch}...')
			# Iteratively train the classifier
			for x_batch, y_batch in zip(x_batches, y_batches):
				self._partial_fit_classifier(
					x_batch, y_batch, learning_rate, beta)
				
	def _partial_fit_decoder(self, x_batch, learning_rate):
		"""
		Train decoder network based on minibatch of input images.

		Parameters
		----------
		x_batch : array-like, shape = [batch_size, height, width, channels]
			A minibatch of input images.
		"""
		feed_dict = {
			self.x_input: x_batch,
			self.learning_rate: learning_rate,
		}
		_ = self.sess.run(self.decoder_optimizer, feed_dict=feed_dict)
        
        step = self.sess.run(self.global_step)
		summary_str = self.sess.run(self.summary_ops_merged, feed_dict=feed_dict)
		self.summary_writer.add_summary(summary_str, step)
	
	def fit_decoder(self, x, num_epochs=5, batch_size=256, learning_rate=1e-3):
		"""
		Train decoder network.

		Parameters
		----------
		x : array-like, shape = [num_samples, height, width, channels]
			A set of input images.
		"""

		# Shuffle x
		num_samples = len(x)
		for epoch in range(num_epochs):
			random_indices = np.random.permutation(num_samples)
			x = x[random_indices]

			# Split x into batches
			num_batches = num_samples // batch_size
			indices = [[k, k+batch_size] for k in range(0, num_samples, batch_size)]
			indices[-1][-1] = num_samples
			x_batches = [x[start:end] for start, end in indices]
			
			print(f'Training epoch {epoch}...')
			# Iteratively train the decoder
			for x_batch in x_batches:
				self._partial_fit_decoder(x_batch, learning_rate)
				
	def predict(self, x):
		"""
		Given a minibatch of input images, predict classes.

		Parameters
		==========
		x : array-like, shape = [batch_size, height, width, channels]
			A minibatch of input images.

		Returns
		=======
		predictions : array, shape = [batch_size, num_classes]
			A matrix of <batch_size> predictive distributions.
		"""

		feed_dict = {
			self.x_input: x,
		}
		predictions = self.sess.run(self.y_pred_denoised, feed_dict=feed_dict)
		return predictions
	
	def compress(self, x):
		"""
		Given a minibatch of input images, create a minibatch of 
		latent means.

		Parameters
		==========
		x : array-like, shape = [batch_size, height, width, channels]
			A minibatch of input images.

		Returns
		=======
		latents : array, shape = [batch_size, num_latent_dimensions]
			A minibatch of latent means.
		"""
		feed_dict = {
			self.x_input: x,
		}
		latents = self.sess.run(self.z_mean, feed_dict=feed_dict)
		return latents
	
	def reconstruct(self, x):
		"""
		Given a minibatch of input images, create a minibatch of reconstructions.

		Parameters
		==========
		x : array-like, shape = [batch_size, height, width, channels]
			A minibatch of input images.

		Returns
		=======
		reconstructions : array, shape = [batch_size, height, width, channels]
			A minibatch of reconstructions.
		"""

		feed_dict = {
			self.x_input: x,
		}
		reconstructions = self.sess.run(self.x_out_denoised, feed_dict=feed_dict)
		return reconstructions
	
	def reconstruct_latents(self, z):
		"""
		Given a minibatch of latent means, create a minibatch of reconstructions.

		Parameters
		==========
		z : array-like, shape = [batch_size, num_latent_dimensions]
			A minibatch of latent means.

		Returns
		=======
		reconstructions : array, shape = [batch_size, height, width, channels]
			A minibatch of reconstructions.
		"""

		feed_dict = {
			self.z_mean: z,
		}
		reconstructions = self.sess.run(self.x_out_denoised, feed_dict=feed_dict)
		return reconstructions