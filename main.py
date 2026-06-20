import tensorflow as tf
import numpy as np

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

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# Evaluate the model

loss, accuracy = model.evaluate(x_test, y_test)
print("Test Accuracy:", accuracy)


# Make a prediction

prediction = model.predict(np.array([x_test[0]]))

print("Predicted Digit:", np.argmax(prediction))
print("Actual Digit:", y_test[0])


# Display the image

plt.imshow(x_test[0], cmap='gray')
plt.show()



