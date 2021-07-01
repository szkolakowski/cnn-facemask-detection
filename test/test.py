try:
	import os
	import numpy as np
	import tensorflow as tf
	from tensorflow import keras
	from tensorflow.keras.preprocessing import image
	import matplotlib.pyplot as plt
except ModuleNotFoundError:
	print('ERROR: Some of the required modules are missing')
	exit()

types = ['mask', 'no mask']

for root, dirs, files in os.walk('../kaggle/test'):
	for file in files:
		path = os.path.join(root, file)
		test = image.load_img(path, target_size=(160,160))
		# hold original image to show with pyplot
		holder = test
		test = image.img_to_array(test)
		test = np.expand_dims(test, axis=0)

		try:
			model = keras.models.load_model('../train/facemask.h5')
			result = model.predict(test)
			result = list(result[0])
			result = result.index(max(result))
		except OSError:
			print('ERROR: CNN Model not found')
			exit()

		plt.grid(False)
		plt.imshow(holder)
		plt.title('Prediction: ' + types[result])
		plt.show()