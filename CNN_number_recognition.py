import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical


# Load data from keras.datasets
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Data structure
print("train_images.shape:", train_images.shape)
print("len(train_labels):", len(train_labels))
print(train_labels)
print("test_images.shape:", test_images.shape)
print("len(test_labels):", len(test_labels))
print(test_labels)

# Building the CNN model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.summary()

# Data formating
train_images, test_images = train_images / 255.0, test_images / 255.0
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Model training
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=5, batch_size=128)

# Testing
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('test_acc:', test_acc)

n_values = 10
predict = model.predict(test_images)
predict = np.eye(n_values, dtype=int)[np.argmax(predict, axis=1)]

index_mistakes = []
for i in range(len(predict)):
    if not np.array_equal(predict[i], test_labels[i]):
        index_mistakes.append(i)

# Summary of incorrectly recognized images
print("len(index_mistakes):", len(index_mistakes))
print("test_acc:", test_acc)
print("Percentage of incorrent predictions by network: ", 100 - test_acc * 100)
print("Percentage of incorrent predictions by index mistakes: ", len(index_mistakes) / len(test_labels) * 100)

# Showing the images using matplotlib
for i in range(0, 10):
    plt.figure(i)
    plt.title(f"Image number: {index_mistakes[i]}. Neural network classified it like "
              f"{np.argmax(predict[index_mistakes[i]])}, should be {np.argmax(test_labels[index_mistakes[i]])}.")
    plt.gray()
    plt.imshow(test_images[index_mistakes[i]].reshape(28, 28))
    plt.show()
