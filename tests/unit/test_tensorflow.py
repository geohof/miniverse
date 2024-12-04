import tensorflow as tf


def test_gpu():
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))




