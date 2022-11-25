import os

import tensorflow as tf
import cv2 as cv
import models
from agents import ClassifierAgent
import numpy as np

print(tf.version.VERSION)

# Define a simple sequential model


classifier = ClassifierAgent(numAgents=4, windowLen=2)


# obtain dataset and normalize
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

numberOfImages=2000

train_labels = train_labels[:numberOfImages]
test_labels = test_labels[:numberOfImages]
train_images = train_images[:numberOfImages] / 255.0
test_images = test_images[:numberOfImages] / 255.0


# Create new train and test arrays and fragment MNIST Dataset
sampledTrainImgs = []
sampleTestImgs = []
trainLabels = []
testLabels = []
for i in range(numberOfImages):

  for j in range(10):
    sampledTrainImgs.append(classifier.sampleImage(train_images[i]))
    sampleTestImgs.append(classifier.sampleImage(test_images[i]))
    trainLabels.append(train_labels[i])
    testLabels.append(test_labels[i])

  train_images[i] = classifier.sampleImage(train_images[i])
  test_images[i] = classifier.sampleImage(test_images[i])

sampledTrainImgs = np.array(sampledTrainImgs)
sampleTestImgs = np.array(sampleTestImgs)
trainLabels = np.array(trainLabels)
testLabels = np.array(testLabels)

#cv.imshow("Image", train_images[0])
print(len(sampledTrainImgs))
#cv.waitKey()

print(sampledTrainImgs[0].shape)
cv.waitKey()

# Create a basic model instance
model = classifier.model

# Display the model's architecture
model.summary()

#checkpoint_path = "training/cp.ckpt"
#checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=models.checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

# Train the model with the new callback
model.fit(sampledTrainImgs, 
          trainLabels,  
          epochs=20,
          validation_data=(sampleTestImgs, testLabels),
          callbacks=[cp_callback])  # Pass callback to training


test_loss, test_acc = model.evaluate(sampleTestImgs, testLabels, verbose=2)
print('\nTest accuracy:', test_acc)




