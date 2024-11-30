import logging
import numpy as np
import pandas as pd
import tensorflow as tf

from miniverse.bin_converter import BinConverter
from miniverse.minimal_miniverse import Miniverse
from miniverse.bug import Bug

from miniverse.timer import Timer

logger = logging.getLogger(__name__)


def test_train_output1():
    bug_spec = {
        'zebra': Bug(pattern='01', repeat=16, frequency=.2),
        'snake': Bug(pattern='0101', segment_len=8, frequency=.3)
    }
    mv_train = Miniverse(bug_spec=bug_spec, verse_len=100000)
    x_train, y_train = mv_train.get_data()

    with Timer(logger=logger, name='set up model'):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.495),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.495),
            tf.keras.layers.Dense(2)
        ])
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    with Timer(logger=logger, name='compile model'):
        model.compile(optimizer='adam',
                      loss=loss_fn,
                      metrics=['accuracy'])
    with Timer(logger=logger, name='train model'):
        model.fit(x_train, y_train, epochs=8)
    with Timer(logger=logger, name='evaluate model'):
        model.evaluate(x_train, y_train, verbose=2)

    mv_test = Miniverse(bug_spec=bug_spec, verse_len=1041)
    x_test, y_test = mv_test.get_data(ordered=True)

    y = model.predict(x_test)
    df = pd.DataFrame(np.hstack([x_test, y]))
    df.to_clipboard(index=False)
    df.to_csv('out1.csv', index=False)


def test_train_output2():
    record_output_len = 4
    record_input_len = 40
    verse_len = 5_000

    bug_spec = {
        'zebra': Bug(pattern='01', repeat=16, frequency=.1),
        'snake': Bug(pattern='0101', segment_len=8, frequency=.2),
        'lion': Bug(pattern='0110', segment_len=8, frequency=.2)
    }
    mv_train = Miniverse(bug_spec=bug_spec, verse_len=verse_len, record_output_len=record_output_len,
                         record_input_len=record_input_len)
    x_train, y_train = mv_train.get_data()
    bc = BinConverter(num_digits=record_output_len)
    y_train = np.array([bc.to_int(''.join(l.astype(str))) for l in y_train])

    with Timer(logger=logger, name='set up model'):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.495),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.495),
            tf.keras.layers.Dense(2 ** record_output_len),
            tf.keras.layers.Softmax()
        ])
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    with Timer(logger=logger, name='compile model'):
        model.compile(optimizer='adam',
                      loss=loss_fn,
                      metrics=['accuracy'])
    with Timer(logger=logger, name='train model'):
        model.fit(x_train, y_train, epochs=8)
    with Timer(logger=logger, name='evaluate model'):
        model.evaluate(x_train, y_train, verbose=2)

    mv_test = Miniverse(bug_spec=bug_spec, verse_len=1041)
    x_test, y_test = mv_test.get_data(ordered=True)

    y = model.predict(x_test)
    df = pd.DataFrame(np.hstack([x_test, y]))
    df.columns = (
            [f'i{i}' for i in range(record_input_len)] + [f'o{bc.to_str(i)}' for i in range(2 ** record_output_len)])
    df.to_clipboard(index=False)
    df.to_csv('out1.csv', index=False)
    o = np.hstack([x_test[:-1], y])
    print(o.transpose())
