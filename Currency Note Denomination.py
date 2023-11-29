#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Conv2D

import numpy as np
from glob import glob


# In[ ]:





# In[2]:


batch = 32
img_height = 180
img_width = 180

train_path = "C:\\Users\\LENOVO\\Desktop\\Note Currency\\Train"
test_path = "C:\\Users\\LENOVO\\Desktop\\Note Currency\\Test"


# In[3]:


train_ds = tf.keras.preprocessing.image_dataset_from_directory(
                train_path,
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch)


# In[4]:


test_ds = tf.keras.preprocessing.image_dataset_from_directory(
  test_path,
#  validation_split=0.1,
 # subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch)


# In[5]:


class_names = train_ds.class_names
print(class_names)


# In[6]:


#Data Visualization

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")


# In[7]:


AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)


# In[8]:


#normalize data by using a rescaling layer

normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
print(np.min(first_image), np.max(first_image))


# In[9]:


Classifier=Sequential()

Classifier.add(layers.experimental.preprocessing.Rescaling(1./255,
      input_shape=(img_height, img_width, 3)))

Classifier.add(Conv2D(32,(3,3),
                      input_shape=(img_height,img_width,3), 
                      activation='relu'))
Classifier.add(MaxPooling2D(pool_size=(2,2)))

Classifier.add(Conv2D(32,(3,3),activation='relu'))
Classifier.add(MaxPooling2D(pool_size=(2,2)))

Classifier.add(Flatten())

Classifier.add(Dense(units = 128, activation = 'relu'))
Classifier.add(Dense(units = 7, activation = 'softmax'))


# In[ ]:





# In[10]:


Classifier.compile(
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  optimizer='adam',
  metrics=['accuracy']
)


# In[11]:


Classifier.summary()


# In[21]:


from PIL import _imaging
from PIL import Image
#fit the classifier
epochs = 3
r = Classifier.fit(
  train_ds,
  validation_data=test_ds,
  epochs=epochs,
)


# In[22]:


mean_val = (sum(r.history['val_accuracy']))/(len(r.history['val_accuracy']))
print('mean_test_accuracy: ', mean_val)
mean_train = (sum(r.history['accuracy']))/(len(r.history['accuracy']))
print('mean_train_accuracy: ', mean_train)


# In[23]:


import matplotlib.pyplot as plt
# plot the loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='test loss')
plt.legend()
plt.show()

# plot the accuracy
plt.plot(r.history['accuracy'], label='train accuracy')
plt.plot(r.history['val_accuracy'], label='test accuracy')
plt.legend()
plt.show()


# In[24]:


#Data Augmentation to reduce overfitting

data_augmentation = tf.keras.Sequential(
  [
    layers.experimental.preprocessing.RandomFlip("horizontal", 
                                                 input_shape=(img_height, 
                                                              img_width,
                                                              3)),
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomZoom(0.1),
  ]
)


# In[25]:


Classifier=Sequential()

Classifier.add(data_augmentation)
Classifier.add(layers.experimental.preprocessing.Rescaling(1./255,
      input_shape=(img_height, img_width, 3)))

Classifier.add(Conv2D(32,(3,3),
                      input_shape=(img_height,img_width,3), 
                      activation='relu'))
Classifier.add(MaxPooling2D(pool_size=(2,2)))

Classifier.add(Conv2D(32,(3,3),activation='relu'))
Classifier.add(MaxPooling2D(pool_size=(2,2)))
Classifier.add(layers.Dropout(0.2))

Classifier.add(Flatten())

Classifier.add(Dense(units = 128, activation = 'relu'))
Classifier.add(Dense(units = 7, activation = 'softmax'))


# In[26]:


Classifier.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


# In[27]:


Classifier.summary()


# In[28]:


epochs = 50
r = Classifier.fit(
  train_ds,
  validation_data=test_ds,
  epochs=epochs
)


# In[29]:


mean_val = (sum(r.history['val_accuracy']))/(len(r.history['val_accuracy']))
print('mean_test_accuracy: ', mean_val)
print('\n')
mean_train = (sum(r.history['accuracy']))/(len(r.history['accuracy']))
print('mean_train_accuracy: ', mean_train)


# In[30]:


import matplotlib.pyplot as plt
# plot the loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='test loss')
plt.legend()
plt.show()

# plot the accuracy
plt.plot(r.history['accuracy'], label='train accuracy')
plt.plot(r.history['val_accuracy'], label='test accuracy')
plt.legend()
plt.show()


# In[41]:


#Use classifier to run prediction on any new image

from tensorflow.keras.preprocessing import image
img = image.load_img(r"C:\Users\LENOVO\Downloads\2000_rupees_note_1684658153455_1684658153652.jpg",
                       target_size=(img_height,img_width))
img


# In[42]:


test_image=image.img_to_array(img)
test_image=np.expand_dims(test_image, axis = 0)

result =np.argmax(Classifier.predict(test_image), axis=1)
result


# In[43]:


prediction = class_names[result[0]]
prediction


# In[44]:


from sklearn.metrics import confusion_matrix, classification_report

# Get the true labels from your test dataset
true_labels = []
for images, labels in test_ds:
    true_labels.extend(labels.numpy())

# Get model predictions for the test dataset
predictions = []
for images, _ in test_ds:
    predictions.extend(Classifier.predict(images).argmax(axis=1))

# Generate the confusion matrix
confusion_mat = confusion_matrix(true_labels, predictions)

# Display the confusion matrix
print("Confusion Matrix:")
print(confusion_mat)

# Generate a classification report
class_report = classification_report(true_labels, predictions, target_names=class_names)

# Display the classification report
print("Classification Report:")
print(class_report)


# In[ ]:





# In[ ]:





# In[45]:


import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Prepare the test data and labels
test_images = []  # Store your test images here
true_labels = []  # Store the true labels of the test images here

for images, labels in test_ds:
    test_images.extend(images)
    true_labels.extend(labels.numpy())

# Convert the test images to a NumPy array
test_images = np.array(test_images)

# Make predictions using your model
predictions = Classifier.predict(test_images)

# Get the predicted labels
predicted_labels = np.argmax(predictions, axis=1)

# Generate the confusion matrix
confusion = confusion_matrix(true_labels, predicted_labels)

# Display the confusion matrix
disp = ConfusionMatrixDisplay(confusion, display_labels=class_names)
disp.plot(cmap='viridis', values_format='d')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




