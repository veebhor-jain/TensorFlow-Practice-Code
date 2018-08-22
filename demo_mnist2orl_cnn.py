from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf
import cv2
import os

currFileDir = os.path.dirname(__file__)  + '/' 

tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  input_layer = tf.reshape(features["x"], [-1, 92, 112, 1])

  # Convolutional Layer #1
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2 and Pooling Layer #2
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Dense Layer
  pool2_flat = tf.reshape(pool2, [-1, 23 * 28 * 64])
  dense = tf.layers.dense(inputs=pool2_flat, units=4096, activation=tf.nn.relu)
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits Layer
  logits = tf.layers.dense(inputs=dropout, units=41)

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
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

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

def load_images_from_folder(folder, n):
    images = []
    filenames = [str(x)+'.pgm' for x in range(n[0], n[1])]
    for filename in filenames:
        img = cv2.imread(os.path.join(folder, filename), 0)
        if img is not None:
            images.append(img.ravel())
    return np.array(images, dtype=np.float32)

def main(unused_argv):
	##### Load training and eval data ####

	# Path to the Dataset and its subfolders
	root_folder = currFileDir + 'orl/'
	sub_folders = []
	for i in range(1, 41):
		sub_folders.extend(['s'+str(i)])
	folders = [os.path.join(root_folder, x) for x in sub_folders]

	# Dividing images into test and train dataset	
	train_data = np.array([img for folder in folders for img in load_images_from_folder(folder, (1, 8))])
	eval_data = np.array([img for folder in folders for img in load_images_from_folder(folder, (8, 11))])

	train_labels = []
	for i in range(1, 41):
		train_labels.extend([i]*7)
	train_labels = np.array(train_labels, dtype = np.int32)
	
	eval_labels = []
	for i in range(1, 41):
		eval_labels.extend([i]*3)
	eval_labels = np.array(eval_labels, dtype = np.int32)
	
	#####

	# Create the Estimator
	mnist_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir="/tmp/orl_con-net_model")

	# Set up logging for predictions
	tensors_to_log = {"probabilities": "softmax_tensor"}
	logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

	# Train the model
	train_input_fn = tf.estimator.inputs.numpy_input_fn(
	    x={"x": train_data},
	    y=train_labels,
	    batch_size=40,
	    num_epochs=None,
	    shuffle=True)
	mnist_classifier.train(
	    input_fn=train_input_fn,
	    steps=280,
	    hooks=[logging_hook])
	
	# Evaluate the model and print results
	eval_input_fn = tf.estimator.inputs.numpy_input_fn(
	    x={"x": eval_data},	
	    y=eval_labels,
	    num_epochs=1,
	    shuffle=False)
	eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
	print(eval_results)

# Our application logic will be added here
if __name__ == "__main__":
  tf.app.run()
