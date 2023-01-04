import tensorflow as tf
print(tf.__version__)
import tensorflow_datasets as tfds


# Construct a tf.data.Dataset
ds = tfds.load('mnist', split='train', shuffle_files=True)