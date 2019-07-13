# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial
import math
import tensorflow as tf

def compute_pairwise_distances(x, y):
  """Computes the squared pairwise Euclidean distances between x and y.
  Args:
    x: a tensor of shape [num_x_samples, num_features]
    y: a tensor of shape [num_y_samples, num_features]
  Returns:
    a distance matrix of dimensions [num_x_samples, num_y_samples].
  Raises:
    ValueError: if the inputs do no matched the specified dimensions.
  """

  if not len(x.get_shape()) == len(y.get_shape()) == 2:
    raise ValueError('Both inputs should be matrices.')

  if x.get_shape().as_list()[1] != y.get_shape().as_list()[1]:
    raise ValueError('The number of features should be the same.')

  norm = lambda x: tf.reduce_sum(tf.square(x), 1)

  # By making the `inner' dimensions of the two matrices equal to 1 using
  # broadcasting then we are essentially substracting every pair of rows
  # of x and y.
  # x will be num_samples x num_features x 1,
  # and y will be 1 x num_features x num_samples (after broadcasting).
  # After the substraction we will get a
  # num_x_samples x num_features x num_y_samples matrix.
  # The resulting dist will be of shape num_y_samples x num_x_samples.
  # and thus we need to transpose it again.
  return tf.transpose(norm(tf.expand_dims(x, 2) - tf.transpose(y)))


def gaussian_kernel_matrix(x, y, sigmas):
  r"""Computes a Guassian Radial Basis Kernel between the samples of x and y.
  We create a sum of multiple gaussian kernels each having a width sigma_i.
  Args:
    x: a tensor of shape [num_samples, num_features]
    y: a tensor of shape [num_samples, num_features]
    sigmas: a tensor of floats which denote the widths of each of the
      gaussians in the kernel.
  Returns:
    A tensor of shape [num_samples{x}, num_samples{y}] with the RBF kernel.
  """
  beta = 1. / (2. * (tf.expand_dims(sigmas, 1)))

  dist = compute_pairwise_distances(x, y)

  s = tf.matmul(beta, tf.reshape(dist, (1, -1)))

  return tf.reshape(tf.reduce_sum(tf.exp(-s), 0), tf.shape(dist))


def maximum_mean_discrepancy(x, y, kernel=gaussian_kernel_matrix):
  r"""Computes the Maximum Mean Discrepancy (MMD) of two samples: x and y.
  Maximum Mean Discrepancy (MMD) is a distance-measure between the samples of
  the distributions of x and y. Here we use the kernel two sample estimate
  using the empirical mean of the two distributions.
  MMD^2(P, Q) = || \E{\phi(x)} - \E{\phi(y)} ||^2
              = \E{ K(x, x) } + \E{ K(y, y) } - 2 \E{ K(x, y) },
  where K = <\phi(x), \phi(y)>,
    is the desired kernel function, in this case a radial basis kernel.
  Args:
      x: a tensor of shape [num_samples, num_features]
      y: a tensor of shape [num_samples, num_features]
      kernel: a function which computes the kernel in MMD. Defaults to the
              GaussianKernelMatrix.
  Returns:
      a scalar denoting the squared maximum mean discrepancy loss.
  """
  with tf.name_scope('MaximumMeanDiscrepancy'):
    # \E{ K(x, x) } + \E{ K(y, y) } - 2 \E{ K(x, y) }
    cost = tf.reduce_mean(kernel(x, x))
    cost += tf.reduce_mean(kernel(y, y))
    cost -= 2 * tf.reduce_mean(kernel(x, y))

    # We do not allow the loss to become negative.
    cost = tf.where(cost > 0, cost, 0, name='value')
  return cost


def mmd_loss(source_samples, target_samples, weight, scope=None):
  """Adds a similarity loss term, the MMD between two representations.
  This Maximum Mean Discrepancy (MMD) loss is calculated with a number of
  different Gaussian kernels.
  Args:
    source_samples: a tensor of shape [num_samples, num_features].
    target_samples: a tensor of shape [num_samples, num_features].
    weight: the weight of the MMD loss.
    scope: optional name scope for summary tags.
  Returns:
    a scalar tensor representing the MMD loss value.
  """
  sigmas = [
      1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100,
      1e3, 1e4, 1e5, 1e6
  ]
  gaussian_kernel = partial(
      gaussian_kernel_matrix, sigmas=tf.constant(sigmas))

  loss_value = maximum_mean_discrepancy(
      source_samples, target_samples, kernel=gaussian_kernel)
  loss_value = tf.maximum(1e-4, loss_value) * weight
  assert_op = tf.Assert(tf.is_finite(loss_value), [loss_value])
  with tf.control_dependencies([assert_op]):
    tag = 'MMD Loss'
    if scope:
      tag = scope + tag
    tf.summary.scalar(tag, loss_value)
    tf.losses.add_loss(loss_value)

  return loss_value


def correlation_loss(source_samples, target_samples, weight, scope=None):
  """Adds a similarity loss term, the correlation between two representations.
  Args:
    source_samples: a tensor of shape [num_samples, num_features]
    target_samples: a tensor of shape [num_samples, num_features]
    weight: a scalar weight for the loss.
    scope: optional name scope for summary tags.
  Returns:
    a scalar tensor representing the correlation loss value.
  """
  with tf.name_scope('corr_loss'):
    source_samples -= tf.reduce_mean(source_samples, 0)
    target_samples -= tf.reduce_mean(target_samples, 0)

    source_samples = tf.nn.l2_normalize(source_samples, 1)
    target_samples = tf.nn.l2_normalize(target_samples, 1)

    source_cov = tf.matmul(tf.transpose(source_samples), source_samples)
    target_cov = tf.matmul(tf.transpose(target_samples), target_samples)

    corr_loss = tf.reduce_mean(tf.square(source_cov - target_cov)) * weight

  assert_op = tf.Assert(tf.is_finite(corr_loss), [corr_loss])
  with tf.control_dependencies([assert_op]):
    tag = 'Correlation Loss'
    if scope:
      tag = scope + tag
    tf.summary.scalar(tag, corr_loss)
    tf.losses.add_loss(corr_loss)

  return corr_loss


def coral_loss(h_src, h_trg, gamma=1e-3):
  # regularized covariances (D-Coral is not regularized actually..)
  # First: subtract the mean from the data matrix
  batch_size = tf.to_float(tf.shape(h_src)[0])
  h_src = h_src - tf.reduce_mean(h_src, axis=0)
  h_trg = h_trg - tf.reduce_mean(h_trg, axis=0)
  cov_source = (1./(batch_size-1)) * tf.matmul(h_src, h_src, transpose_a=True) #+ gamma * tf.eye(self.hidden_repr_size)
  cov_target = (1./(batch_size-1)) * tf.matmul(h_trg, h_trg, transpose_a=True) #+ gamma * tf.eye(self.hidden_repr_size)
  # Returns the Frobenius norm (there is an extra 1/4 in D-Coral actually)
  # The reduce_mean account for the factor 1/d^2
  return tf.reduce_mean(tf.square(tf.subtract(cov_source,cov_target)))


def log_coral_loss(h_src, h_trg, gamma=1e-3):
  # regularized covariances result in inf or nan
  # First: subtract the mean from the data matrix
  batch_size = tf.to_float(tf.shape(h_src)[0])
  h_src = h_src - tf.reduce_mean(h_src, axis=0)
  h_trg = h_trg - tf.reduce_mean(h_trg, axis=0)
  cov_source = (1./(batch_size-1)) * tf.matmul(h_src, h_src, transpose_a=True) #+ gamma * tf.eye(self.hidden_repr_size)
  cov_target = (1./(batch_size-1)) * tf.matmul(h_trg, h_trg, transpose_a=True) #+ gamma * tf.eye(self.hidden_repr_size)
  #eigen decomposition
  eig_source  = tf.self_adjoint_eig(cov_source)
  eig_target  = tf.self_adjoint_eig(cov_target)
  log_cov_source = tf.matmul(eig_source[1], tf.matmul(tf.diag(tf.log(eig_source[0])), eig_source[1], transpose_b=True))
  log_cov_target = tf.matmul(eig_target[1], tf.matmul(tf.diag(tf.log(eig_target[0])), eig_target[1], transpose_b=True))

  # Returns the Frobenius norm
  return tf.reduce_mean(tf.square(tf.subtract(log_cov_source,log_cov_target)))
