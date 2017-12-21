"""
Read in the original Tiny Imagenet dataset and output a tfrecord file of
the entire dataset.
Reference: https://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/
"""

import itertools
import os
import numpy as np
import skimage.io as io
import skimage.color as color
import tensorflow as tf


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def write_to_tfrecord(filename, images, labels):
    writer = tf.python_io.TFRecordWriter(filename)

    for image, label in zip(images, labels):
        image_raw = image.tostring()
        features = tf.train.Features(feature={
            'height': _int64_feature(64),
            'width': _int64_feature(64),
            'image_raw': _bytes_feature(image_raw),
            'label': _int64_feature(label),
        })
        example = tf.train.Example(features=features)
        writer.write(example.SerializeToString())

    writer.close()


def verify_tfrecord(filename, original_images, original_labels):
    reconstructed_images = []
    reconstructed_labels = []

    record_iterator = tf.python_io.tf_record_iterator(path=filename)
    for string_record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(string_record)

        height = int(example.features.feature['height'].int64_list.value[0])
        width = int(example.features.feature['width'].int64_list.value[0])
        img_string = example.features.feature['image_raw'].bytes_list.value[0]
        label = int(example.features.feature['label'].int64_list.value[0])

        img_1d = np.fromstring(img_string, dtype=np.uint8)
        image = np.reshape(img_1d, newshape=(height, width, 3))

        reconstructed_images.append(image)
        reconstructed_labels.append(label)

    reconstructed_images = np.array(reconstructed_images)
    reconstructed_labels = np.array(reconstructed_labels)

    for original_image, original_label, reconstructed_image, reconstructed_label in \
            zip(original_images, original_labels, reconstructed_images, reconstructed_labels):
        assert np.allclose(original_image, reconstructed_image)
        assert np.allclose(original_label, reconstructed_label)


def main():
    cwd = os.getcwd()
    dataset_dir = os.path.join(cwd, 'datasets/tiny-imagenet-200/')
    output_dir = os.path.join(dataset_dir, 'cached/')
    os.makedirs(output_dir, exist_ok=True)

    # Map each annotation to an integer label (e.g, "n03444034": 42)
    annotations_file = os.path.join(dataset_dir, 'wnids.txt')
    with open(annotations_file) as f:
        annotations = f.readlines()
    counter = itertools.count()
    annotation_to_label = {a.strip(): next(counter) for a in annotations}

    train_dir = os.path.join(dataset_dir, 'train/')
    test_dir = os.path.join(dataset_dir, 'val/')

    train_images = []
    train_labels = []

    for annotation in os.listdir(train_dir):
        label = annotation_to_label[annotation]
        images_dir = os.path.join(train_dir, annotation, 'images/')
        for filename in os.listdir(images_dir):
            image_path = os.path.join(images_dir, filename)
            image = io.imread(image_path)
            if image.shape == (64, 64):  # greyscale image
                image = color.grey2rgb(image)
            train_images.append(image)
            train_labels.append(label)

    # Map each test image to its annotation (e.g, "val_12.JPEG": "n02226429")
    test_labels_file = os.path.join(test_dir, 'val_annotations.txt')
    image_to_label = {}
    with open(test_labels_file) as f:
        for line in f:
            filename, annotation = line.split('\t')[0:2]
            image_to_label[filename] = annotation_to_label[annotation]

    test_images = []
    test_labels = []
    test_images_dir = os.path.join(test_dir, 'images/')
    for filename in os.listdir(test_images_dir):
        label = image_to_label[filename]
        image_path = os.path.join(test_images_dir, filename)
        image = io.imread(image_path)
        if image.shape == (64, 64):  # greyscale image
            image = color.grey2rgb(image)
        test_images.append(image)
        test_labels.append(label)

    train_images = np.array(train_images)
    train_labels = np.array(train_labels)

    test_images = np.array(test_images)
    test_labels = np.array(test_labels)

    np.save(os.path.join(output_dir, 'train_images.npy'), train_images)
    np.save(os.path.join(output_dir, 'train_labels.npy'), train_labels)
    np.save(os.path.join(output_dir, 'test_images.npy'), test_images)
    np.save(os.path.join(output_dir, 'test_labels.npy'), test_labels)

    train_tfrecords_filename = os.path.join(output_dir, 'train.tfrecords')
    write_to_tfrecord(train_tfrecords_filename, train_images, train_labels)
    verify_tfrecord(train_tfrecords_filename, train_images, train_labels)

    test_tfrecords_filename = os.path.join(output_dir, 'test.tfrecords')
    write_to_tfrecord(test_tfrecords_filename, test_images, test_labels)
    verify_tfrecord(test_tfrecords_filename, test_images, test_labels)


if __name__ == '__main__':
    main()
