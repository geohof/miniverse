import numpy as np
import pandas as pd

from minimal_miniverse import Miniverse
import tensorflow as tf


def test_create_miniverse():
    mv = Miniverse()
    mvv = mv.verse
    assert len(mvv) == mv.verse_len
    assert set(mvv).issubset({0, 1})


def test_train():
    mv = Miniverse()
    mvd = mv.get_data()
    mvdl = list(mvd)
    xtrain = 0.99 * np.array(mvdl[0])
    ytrain = np.array(mvdl[1])

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.495),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.495),
        tf.keras.layers.Dense(2)
    ])
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['accuracy'])
    model.fit(xtrain, ytrain, epochs=5)
    model.evaluate(0.5 * np.array(mvdl[0]),  0.5 * np.array(mvdl[1]), verbose=2)
    test_x = xtrain[0:1000]
    y = model.predict(test_x)
    df = pd.DataFrame(np.hstack([test_x, y]))
    df.to_clipboard(index=False)
    df.to_csv('out.csv', index=False)




def test_tutorial():
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print(f'{type(x_train) = }')
    print(f'{x_train[0] = }')
    print(f'{y_train[0] = }')

