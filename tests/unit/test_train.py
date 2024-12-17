import logging
import numpy as np
import tensorflow as tf
from pathlib import Path
from pytest import mark

from miniverse.bin_converter import BinConverter
from miniverse.miniverse import Miniverse
from miniverse.animal import Animal

from miniverse.timer import Timer

FILE_DIRECTORY = Path(__file__).parent

logger = logging.getLogger(__name__)

zoo = {
    'zebra': Animal(pattern='01', repeat=16, frequency=.1),
    'snake': Animal(pattern='0101', segment_len=8, frequency=.2),
    'lion': Animal(pattern='0110', segment_len=8, frequency=.2)
}

record_output_len = 1
record_input_len = 40
verse_len = 500_000
model_directory = FILE_DIRECTORY.parent.parent / 'models'
model_directory.mkdir(exist_ok=True)
id_str = f'{verse_len}_{record_output_len}'
model_file_name = model_directory / f'model_{id_str}.keras'
output_file_name = model_directory / f'output_{id_str}.csv'


@mark.skipif(model_file_name.exists(), reason='model already exists')
def test_train_output():
    mv_train = Miniverse(zoo=zoo, verse_len=verse_len, record_output_len=record_output_len,
                         record_input_len=record_input_len, random_seed=606)
    x_train, y_train = mv_train.get_training_data()
    bc = BinConverter(num_digits=record_output_len)

    y_train = np.array([bc.to_int(''.join(l.astype(str))) for l in y_train])

    with Timer(logger=logger, name='set up model'):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(2 ** record_output_len),
            tf.keras.layers.Softmax()
        ])
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    with Timer(logger=logger, name='compile model'):
        model.compile(optimizer='adam',
                      loss=loss_fn,
                      metrics=['accuracy'])
    with Timer(logger=logger, name='train model'):
        model.fit(x_train, y_train, epochs=5)

    model.save(model_file_name)
def test_eval():
    model = tf.keras.models.load_model(model_file_name)
    mv_test = Miniverse(zoo={'snake': Animal(pattern='0101', segment_len=8, frequency=.4)},
                        verse_len=120, animal_position='equidistant', record_output_len=record_output_len,
                        record_input_len=record_input_len, random_seed=606)
    x_test = mv_test.get_input_data()
    mv_test.prediction = model.predict(x_test)
    mv_test.prediction_to_plt(to_index=42).show()
    df = mv_test.prediction_to_data_frame(to_index=42)
    df.to_csv(output_file_name, index=False)
    print(*mv_test.prediction_to_strs(to_index=42), sep="\n")
