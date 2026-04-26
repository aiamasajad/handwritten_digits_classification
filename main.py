import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load the MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# Build the model
model = tf.keras.models.Sequential([                   #model is built layer by layer
    tf.keras.layers.Flatten(input_shape=(28,28)),      #FLATTEN -> input layer is 2D matrix which will get converted to 1D vector (nn need data in single line
    tf.keras.layers.Dense(128, activation='relu'),    #DENSE -> Hidden layer 
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])


# Train the model
history = model.fit(x_train,y_train, epochs=4, validation_data=(x_test,y_test))

# Plot training & validation accuracy values
img = cv2.imread("digit_8.png")[:,:,0]
img = np.invert(np.array([img]))
prediction = model.predict(img)
print("Predicted digit:", np.argmax(prediction))
plt.imshow(img[0], cmap=plt.cm.binary)
plt.show()
