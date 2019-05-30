#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#	http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Convolutional Neural Network Estimator for MNIST, built with tf.layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import time
import datetime
import json
#from my_minist import MNIST_W
import os
starttime = datetime.datetime.now()
tf.logging.set_verbosity(tf.logging.INFO)
#import pdb
#wjm_output="/home/output"
#wjm_steps=1000
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("TRAIN_DATA","/opt/ml/input/data/training/cifar10_train.tfrecord",'input data')
tf.app.flags.DEFINE_string("EVAL_DATA","/opt/ml/input/data/validation/cifar10_test.tfrecord",'test data')
tf.app.flags.DEFINE_string("EXPORT_NAME","cifar10",'export name')
tf.app.flags.DEFINE_string("EVAL_NAME","cifar10",'eval name')
tf.app.flags.DEFINE_string("MODEL_DIR","/opt/ml/model",'output dir folder')

tf.app.flags.DEFINE_integer("MAX_STEPS",10000,'train max steps')
tf.app.flags.DEFINE_integer("TRAIN_BATCH_SIZE",100,'tain batch size')
tf.app.flags.DEFINE_integer("EVAL_STEPS",1000,'eval steps')
tf.app.flags.DEFINE_integer("EVAL_BATCH_SIZE",100,'eval batch size')

def read_and_decode(serialized_example):
	keys_to_features = {
	    'data': tf.FixedLenFeature([], tf.string, default_value=''),
	    'format': tf.FixedLenFeature([], tf.string, default_value='png'),
	    'label': tf.FixedLenFeature([], tf.int64, default_value=tf.zeros([], dtype=tf.int64))
	}
	
	features = tf.parse_single_example(serialized_example, keys_to_features)
	image = tf.decode_raw(features['data'], tf.uint8)
	image = tf.cast(image, tf.float32)
	image = tf.reshape(image, [32, 32, 3])
	image = tf.image.per_image_standardization(image)
	label = features['label']
	#label = tf.one_hot(label, 10, 1, 0)
	return image, label


def input_fn(filenames, num_epochs=None, shuffle=True, skip_header_lines=0, batch_size=200, num_parallel_calls=None, prefetch_buffer_size=None):
	dataset = tf.data.TFRecordDataset(filenames)
	dataset = dataset.map(read_and_decode)
	iterator = dataset.repeat(num_epochs).batch(batch_size).make_one_shot_iterator()
	image, label = iterator.get_next()
	return image,label

def train_input():
	return input_fn(FLAGS.TRAIN_DATA,batch_size=FLAGS.TRAIN_BATCH_SIZE)

def eval_input():
	return input_fn(FLAGS.EVAL_DATA,batch_size=FLAGS.EVAL_BATCH_SIZE)

def cnn_model_fn(features, labels, mode):
	"""Model function for CNN."""
	# Input Layer
	# Reshape X to 4-D tensor: [batch_size, width, height, channels]
	# MNIST images are 28x28 pixels, and have one color channel
	#input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])
	input_layer = tf.reshape(features, [-1, 32, 32, 3])
	
	# Convolutional Layer #1
	# Computes 32 features using a 5x5 filter with ReLU activation.
	# Padding is added to preserve width and height.
	# Input Tensor Shape: [batch_size, 28, 28, 1]
	# Output Tensor Shape: [batch_size, 28, 28, 32]
	# Input Tensor Shape: [batch_size, 32, 32, 3]
	# Output Tensor Shape: [batch_size, 32, 32, 32]
	conv1 = tf.layers.conv2d(
		inputs=input_layer,
		filters=32,
		kernel_size=[5, 5],
		padding="same",
		activation=tf.nn.relu)

	# Pooling Layer #1
	# First max pooling layer with a 2x2 filter and stride of 2
	# Input Tensor Shape: [batch_size, 28, 28, 32]
	# Output Tensor Shape: [batch_size, 14, 14, 32]
	# Input Tensor Shape: [batch_size, 32, 32, 32]
	# Output Tensor Shape: [batch_size, 16, 16, 32]
	pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
	
	# Convolutional Layer #2
	# Computes 64 features using a 5x5 filter.
	# Padding is added to preserve width and height.
	# Input Tensor Shape: [batch_size, 14, 14, 32]
	# Output Tensor Shape: [batch_size, 14, 14, 64]
	# Input Tensor Shape: [batch_size, 16, 16, 32]
	# Output Tensor Shape: [batch_size, 16, 16, 64]
	conv2 = tf.layers.conv2d(
		inputs=pool1,
		filters=64,
		kernel_size=[5, 5],
		padding="same",
		activation=tf.nn.relu)

	# Pooling Layer #2
	# Second max pooling layer with a 2x2 filter and stride of 2
	# Input Tensor Shape: [batch_size, 14, 14, 64]
	# Output Tensor Shape: [batch_size, 7, 7, 64]
	# Input Tensor Shape: [batch_size, 16, 16, 64]
	# Output Tensor Shape: [batch_size, 8, 8, 64]
	pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
	
	# Flatten tensor into a batch of vectors
	# Input Tensor Shape: [batch_size, 7, 7, 64]
	# Output Tensor Shape: [batch_size, 7 * 7 * 64]
	# Input Tensor Shape: [batch_size, 8, 8, 64]
	# Output Tensor Shape: [batch_size, 8 * 8 * 64]
	pool2_flat = tf.reshape(pool2, [-1, 8 * 8 * 64])#*****
	
	# Dense Layer
	# Densely connected layer with 1024 neurons
	# Input Tensor Shape: [batch_size, 7 * 7 * 64]
	# Output Tensor Shape: [batch_size, 1024]
	dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
	
	# Add dropout operation; 0.6 probability that element will be kept
	dropout = tf.layers.dropout(
		inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

	# Logits layer
	# Input Tensor Shape: [batch_size, 1024]
	# Output Tensor Shape: [batch_size, 10]
	logits = tf.layers.dense(inputs=dropout, units=10)

	predictions = {
		# Generate predictions (for PREDICT and EVAL mode)
		"classes": tf.argmax(input=logits, axis=1),
		# Add `softmax_tensor` to the graph. It is used for PREDICT and by the
		# `logging_hook`.
		"probabilities": tf.nn.softmax(logits, name="softmax_tensor")
	}
	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

	# Calculate Loss (for both TRAIN and EVAL modes)
	loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)#oneshot , number_lable

  # Configure the Training Op (for TRAIN mode)
	if mode == tf.estimator.ModeKeys.TRAIN:
		optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
		train_op = optimizer.minimize(
			loss=loss,
			global_step=tf.train.get_global_step())
		return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
	eval_metric_ops = {
		"accuracy": tf.metrics.accuracy(
		labels=labels, predictions=predictions["classes"])} 
	return tf.estimator.EstimatorSpec(
		mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def _get_session_config_from_env_var():
	tf_config = json.loads(os.environ.get('TF_CONFIG', '{}'))
	
	if (tf_config and 'task' in tf_config and 'type' in tf_config['task'] and
		'index' in tf_config['task']):
	    # Master should only communicate with itself and ps
		if tf_config['task']['type'] == 'master':
			return tf.ConfigProto(device_filters=['/job:ps', '/job:master'])
	    # Worker should only communicate with itself and ps
		elif tf_config['task']['type'] == 'worker':
			return tf.ConfigProto(device_filters=[
				'/job:ps',
				'/job:worker/task:%d' % tf_config['task']['index']
			])
	return None


def main(unused_argv):
	
		
	#pdb.set_trace()
	# Create the Estimator
	
	# Set up logging for predictions
	# Log the values in the "Softmax" tensor with label "probabilities"
	#tensors_to_log = {"probabilities": "softmax_tensor"}
	#logging_hook = tf.train.LoggingTensorHook(
	#tensors=tensors_to_log, every_n_iter=50)


	# Train the model
	train_spec = tf.estimator.TrainSpec(train_input, max_steps=FLAGS.MAX_STEPS)
	eval_spec = tf.estimator.EvalSpec(eval_input, steps=FLAGS.EVAL_STEPS, name=FLAGS.EVAL_NAME)
	run_config = tf.estimator.RunConfig(session_config=_get_session_config_from_env_var())
	run_config = run_config.replace(model_dir=FLAGS.MODEL_DIR)

	cifar10_classifier = tf.estimator.Estimator(
		model_fn=cnn_model_fn, config=run_config)

	tf.estimator.train_and_evaluate(cifar10_classifier, train_spec, eval_spec)

	# Train the model
	#train_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
	#	x={"x": train_data},
	#	y=train_labels,
	#	batch_size=100,
	#	num_epochs=None,
	#shuffle=True)
	#mnist_classifier.train(
#		input_fn=train_input_fn,
#		steps=wjm_steps,
#	hooks=[logging_hook])

	# Evaluate the model and print results
	#eval_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
	#	x={"x": eval_data}, y=eval_labels, num_epochs=1, shuffle=False)
	#eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
	#tf.logging.info("end")
	tf.logging.info(time.strftime("%Y.%m.%d:%H.%M.%S",time.localtime(time.time())))
	endtime = datetime.datetime.now()
	last_time=(endtime-starttime).seconds
	tf.logging.info(last_time)
	#os.system("echo %s > /opt/ai/output/a.txt"%str(last_time))
	#print(eval_results)

if __name__ == "__main__":
	tf.logging.info("start")
	tf.logging.info(time.strftime("%Y.%m.%d:%H.%M.%S",time.localtime(time.time())))
	tf.app.run()
