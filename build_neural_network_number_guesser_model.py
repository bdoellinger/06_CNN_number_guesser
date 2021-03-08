import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import random

# model / data parameters
num_classes = 10                            # numbers from 0 to 9
img_width, img_height = 28, 28              # picture dimensions of training data
input_shape = (img_width, img_height, 1)
batch_size = 100
epochs = 15
validation_split = 0.2
verbosity = 1

# mnist data (70 000 images of numbers from 0 to 9), split between train and test sets (60 000 train, 10 000 test samples)
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")
# adjust data that the values of x components are in the range [0,1] 
x_train, x_test = x_train / 255.0, x_test / 255.0
# adjust images to shape (28, 28, 1) instead of (28,28)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print("image shape:", x_train[0].shape)
# convert class vectors to binary class matrices, i.e. x_train[0] is an image of the number 5, hence the label y_train[0] is 5 which will be converted to [0 0 0 0 0 1 0 0 0 0] (5-th indexed element of array is 1 else 0)
print(y_train[0])
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
print(y_train[0])
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

# build model with 3x3 matrix kernels, as explained in keras.io
model = tf.keras.Sequential(
    [
        tf.keras.Input(shape=input_shape),
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ]
)

# to get details about model
model.summary()

# train model with features x_train and labels y_train
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs, validation_split=validation_split)
score = model.evaluate(x_test, y_test, verbose=verbosity)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

# test model on 10 random examples
samples = [random.randint(0,6000) for _ in range(10)]
samples_to_predict = []

# Generate plots for samples
for sample in samples:
  # Generate a plot
  reshaped_image = x_train[sample].reshape((img_width, img_height))
  plt.imshow(reshaped_image)
  plt.show()
  # Add sample to array for prediction
  samples_to_predict.append((x_train[sample],y_train[sample]))

# function that returns index, where max in arr is first attained
def max_index(arr):
  max_ind, max_val = -1, float("-Inf")
  for index, value in enumerate(arr):
    if value > max_val:
      max_ind, max_val = index, value
  return max_ind

# Generate predictions for samples
for x_sample, y_sample in samples_to_predict:
  prediction = model.predict(np.array([x_sample]))
  print("__________________________________________________")
  print("")
  print(prediction)
  print("")
  print("prediction: " + str(max_index(prediction[0])) + " actual lable: " + str(max_index(y_sample)))
print("__________________________________________________")

# save model with trained weights
model.save('tensorflow_CNN_number_guesser_model')


