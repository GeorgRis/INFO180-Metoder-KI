import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm import tqdm

# Keras is an open-source library that provides a Python interface for artificial neural networks.
# Keras acts as an interface for the TensorFlow library.
# You can load some of the datasets including fashion-mnist using Keras.

mnist = tf.keras.datasets.fashion_mnist

# Split the data into train and test
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
# Normalizing the data by scaling pixels in the range 0-1 (divide values by 255 since pixel values are between 0 and 255)
train_images = (np.expand_dims(train_images, axis=-1)/255.).astype(np.float32)
train_labels = (train_labels).astype(np.int64)
test_images = (np.expand_dims(test_images, axis=-1)/255.).astype(np.float32)
test_labels = (test_labels).astype(np.int64)

plt.imshow(train_images[0])

plt.figure(figsize=(10,10))
random_inds = np.random.choice(60000,25)
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    image_ind = random_inds[i]
    plt.imshow(np.squeeze(train_images[image_ind]), cmap=plt.cm.binary)
    plt.xlabel(train_labels[image_ind])

def build_fc_model():
  # Model is defined using sequential class from Keras
  fc_model = tf.keras.Sequential([
      # First define a Flatten layer, we need this to prepare the input in a flattened form (rather than having 28x28 pixels, we have 784 pixels)
      tf.keras.layers.Flatten(),

      # '''TOCOMPLETE: Pick the activation function for the first fully connected (Dense) layer.'''
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(10, activation='softmax')


  ])
  return fc_model

model = build_fc_model()

# '''TOCOMPLETE: Try different optimizers and learning rates. How do these affect
#    the accuracy of the trained model? Which optimizers and/or learning rates gives the best performance?'''
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=1e-1),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Pick the batch size and the number of epochs to use during training
BATCH_SIZE = 64
EPOCHS = 10

model.fit(train_images, train_labels, batch_size=BATCH_SIZE, epochs=EPOCHS)

# '''TOCOMPLETE: The evaluate method is used to test the model!'''
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('Test accuracy:', test_acc)

def build_cnn_model():
    cnn_model = tf.keras.Sequential([

        # TOCOMPLETE: Define the first convolutional layer
        tf.keras.layers.Conv2D(2, (3, 3), activation='relu', input_shape=(28, 28, 1)),

        # TOCOMPLETE: Define the first max pooling layer
        tf.keras.layers.MaxPool2D((2, 2)),

        # TOCOMPLETE: Define the second convolutional layer
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),

        # TOCOMPLETE: Define the second max pooling layer
        tf.keras.layers.MaxPool2D((2,2)),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),

        # TOCOMPLETE: Define the last Dense layer to output the classification probabilities.
        # Pay attention to the activation needed a probability output.
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    return cnn_model

cnn_model = build_cnn_model()
# Initialize the model by passing some data through
cnn_model.predict(train_images[[0]])
# Print the summary of the layers in the model.
print(cnn_model.summary())

# '''TOCOMPLETE: The compile operation needs an optimizer and learning rate of choice'''
cnn_model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# '''TOCOMPLETE: Use model.fit to train the CNN model, with the same batch_size and number of epochs previously used.'''
cnn_model.fit(train_images, train_labels, epochs=10,
                    validation_data=(test_images, test_labels))

# '''TOCOMPLETE: The evaluate method is used to test the model!'''
test_loss, test_acc =  model.evaluate(test_images,  test_labels, verbose=2)

print('Test accuracy:', test_acc)
predictions = cnn_model.predict(test_images)
score = tf.nn.softmax(predictions[3])

print(predictions[3])
# '''TOCOMPLETE: Find the item with the highest confidence prediction for the third image in the testset.'''
'''prediction = max(cnn_model.predict(test_images))'''

print('This image most likely belongs to {} with a {:.2f} percent confidence.'.format(
        class_names[np.argmax(score)], 100 * np.max(score)))


print("The true label for this item is:", test_labels[3])
plt.imshow(test_images[3,:,:,0], cmap=plt.cm.binary)
