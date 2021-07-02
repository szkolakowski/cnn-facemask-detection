### cnn-brain-tumor
##### About
CNN Brain Tumor is a classic convolutional neural network, specifically trained for recognizing brain tumors on skull tomography scans. With prediction accuracy
of 94.74% it is rarely mistaken. In order to test the accuracy of the trained model, simply put some clean skull tomography scans (no additional text and arrows,
only bright skull on dark background) in `test/data` directory of the project. Script will show all images with calculated prediction written above. Projects uses
`try:` and `except:` functions to make sure all files are present.
##### Required Modules
In order to start a project you need to create a virtual environment with:\
*tensorflow*\
*keras*\
*numpy*\
*pillow*\
*matplotlib*
##### Kaggle
Kaggle folder contains a database of skull tomography scans with precise description of all files.\
Database found on https://www.kaggle.com/preetviradiya/brian-tumor-dataset
##### Test
Test folder contains `data` directory and `test.py` script.\
If you want to test trained model, simply put some photos in `data` directory and launch `test.py` script.\
The main objective of `test.py` is to extract all test data from the `data` directory, and prepare it for the model. Preparation is all about changing the
pixel width and height of an image, and then transfering it into a *numpy* array. After that, script loads the model from `../train/brain-tumor-model.h5` and
makes predictions with it. Every image will be displayed using *matplotlib.pyplot* with a result of `model.predict`.
##### Train
Train folder contains two python scripts `train.py`, `model.py` and already trained model `brain_tumor_model.h5`.\
Script `train.py` imports images from kaggle database and preprocesses them using `ImageDataGenerator`. Images target size is set to 160x160px. Their batch size
is set to 32. `train.py` uses model builder from `model.py`, giving it the attributes of `output_types, activation, loss, optimizer`. I made a couple of tests,
and a pair of `loss='log_cosh'` with `optimizer=Nadam(learning_rate=0.0003)` gave the best final accuracy and loss after 16 training epochs.
##### Model
Model summary:
Layer (type) | Output shape | Params
| :---: | :---: | :---: |
input_1 (InputLayer) | [(None, 160, 160, 3)] | 0
conv2d (Conv2D) | (None, 160, 160, 16) | 448 
max_pooling2d (MaxPooling2D) | (None, 80, 80, 16) | 0     
conv2d_1 (Conv2D) | (None, 80, 80, 32) | 4640 
max_pooling2d_1 (MaxPooling2D) | (None, 40, 40, 32) | 0 
conv2d_2 (Conv2D) | (None, 40, 40, 64) | 18496   
max_pooling2d_2 (MaxPooling2D) | (None, 20, 20, 64) | 0   
flatten (Flatten) | (None, 25600) | 0    
dense (Dense) | (None, 128) | 3276928   
dense_1 (Dense) | (None, 2) | 258

`Total params: 3,300,770`
`Trainable params: 3,300,770`
`Non-trainable params: 0`
