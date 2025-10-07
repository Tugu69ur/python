import tensorflow as tf
print(tf.__version__)

# Import NumPy - package for working with arrays in Python.
import numpy as np

# Import useful keras functions - this is similar to the
# TensorFlow.js Layers API functionality.
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# Create a new dense layer with 1 unit, and input shape of [1].
layer0 = Dense(units=1, input_shape=[1])
model = Sequential([layer0])

# Compile the model using stochastic gradient descent as optimiser
# and the mean squared error loss function.
model.compile(optimizer='sgd', loss='mean_absolute_error')

# Provide some training data! Here we are using some fictional data 
# for house square footage and house price (which is simply 1000x the 
# square footage) which our model must learn for itself.
xs = np.array([800.0, 850.0, 900.0, 950.0, 980.0, 1000.0, 1050.0, 1075.0, 1100.0, 1150.0, 1200.0, 1250.0, 1300.0, 1400.0, 1500.0, 1600.0, 1700.0, 1800.0, 1900.0, 2000.0], dtype=float)

ys = np.array([800000.0, 850000.0, 900000.0, 950000.0, 980000.0, 1000000.0, 1050000.0, 1075000.0, 1100000.0, 1150000.0, 1200000.0,  1250000.0, 1300000.0, 1400000.0, 1500000.0, 1600000.0, 1700000.0, 1800000.0, 1900000.0, 2000000.0], dtype=float)

# Train the model for 500 epochs.
model.fit(xs, ys, epochs=500, verbose=0)

# Test the trained model on a test input value
print(model.predict([1200.0]))

# Save the model we just trained to the "SavedModel" format to the
# same directory our test.py file is located.
tf.saved_model.save(model, './')