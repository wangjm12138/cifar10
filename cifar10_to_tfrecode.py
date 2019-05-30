import tensorflow as tf
import os
import sys
import pickle
import numpy as np
import pdb
from PIL import Image

_NUM_TRAIN_FILES = 5
LABELS_FILENAME = 'label.txt'

_CLASS_NAMES = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'
]

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def _int64_feature(value):
    if not isinstance(value, (tuple, list)):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
 

def _bytes_feature(value):
	#a = tf.train.BytesList(value=[value])
	#b = tf.train.Feature(bytes_list=a)
	#print(type(a))
	#print(type(b))
	#pdb.set_trace()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _image_to_tfexample(image_data, image_format, class_id):
    return tf.train.Example(features=tf.train.Features(feature={
        'data': _bytes_feature(image_data),
		'format': _bytes_feature(image_format),
        'label': _int64_feature(class_id)
#        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_data])),
#        'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_format])),
#        'image/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=value)),
    }))


def _get_output_filename(split_name):
    return 'cifar10_%s.tfrecord' % (split_name)

def _add_to_tfrecord(filename, tfrecord_writer, offset=0):
    data = unpickle(filename)
    images = data[b'data']
    num_images = images.shape[0]
    images = images.reshape((num_images, 3, 32, 32))
    labels = data[b'labels']
 
    with tf.Graph().as_default():
        for j in range(num_images):
            sys.stdout.write('\r>> Reading file [%s] image %d/%d' % (
                filename, offset + j + 1, offset + num_images))
            sys.stdout.flush()
 
            image = np.squeeze(images[j]).transpose((1, 2, 0))
            image = Image.fromarray(image)
            #image = image.resize((32,32))
            # image.save('../images/image/' + str(j) + '.png')
            image = image.tobytes()
            #print(image)
            label = labels[j]
 
            example = _image_to_tfexample(image, b'png', label)
            tfrecord_writer.write(example.SerializeToString())
 
    return offset + num_images

def run():
 
    training_filename = _get_output_filename('train')
    testing_filename = _get_output_filename('test')
 
    # First, process the training data:
    with tf.python_io.TFRecordWriter(training_filename) as tfrecord_writer:
        offset = 0
        for i in range(_NUM_TRAIN_FILES):
            filename = './data_batch_%s' % (i + 1)  # 1-indexed.
            offset = _add_to_tfrecord(filename, tfrecord_writer, offset)
 
    # Next, process the testing data:
    with tf.python_io.TFRecordWriter(testing_filename) as tfrecord_writer:
        filename = './test_batch'
        _add_to_tfrecord(filename, tfrecord_writer)
 
    # Finally, write the labels file:
    #labels_to_class_names = dict(zip(range(len(_CLASS_NAMES)), _CLASS_NAMES))
    #write_label_file(labels_to_class_names, dataset_dir)
 
    print('\nFinished converting the Cifar10 dataset!')


if __name__ == '__main__':
    run()
