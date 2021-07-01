import tensorflow as tf
from tensorflow import keras

def build(output_types, activation, loss, optimizer):
	data = tf.keras.Input(shape=(160, 160, 3))
	inputs = tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu')(data)
	inputs = tf.keras.layers.MaxPooling2D(strides=2)(inputs)
	inputs = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
	inputs = tf.keras.layers.MaxPooling2D(strides=2)(inputs)
	inputs = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(inputs)
	inputs = tf.keras.layers.MaxPooling2D(strides=2)(inputs)
	inputs = tf.keras.layers.Flatten()(inputs)
	inputs = tf.keras.layers.Dense(128, activation='relu')(inputs)
	outputs = tf.keras.layers.Dense(output_types, activation=activation)(inputs)

	model = tf.keras.Model(inputs=data, outputs=outputs)
	model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

	return model