# EX 4 Deep Neural Network for Malaria Infected Cell Recognition

## AIM

To develop a deep neural network for Malaria infected cell recognition and to analyze the performance.

## Problem Statement and Dataset
To develop a deep neural network to accurately identify malaria-infected cells in microscopic blood images. This automated system should achieve high performance in diagnosis, improve treatment decisions, and potentially be deployed in resource-limited settings.Your task would be to optimize the model, possibly by tuning hyperparameters, trying different architectures, or using techniques like transfer learning to improve classification accuracy.
## Neural Network Model
![image5](https://github.com/user-attachments/assets/28f45165-e5f3-4be5-b0c1-17157f509bbb)

## DESIGN STEPS

### STEP 1:
Import Libraries: Load necessary libraries for data processing, visualization, and building the neural network.
### STEP 2:
Configure GPU: Set up TensorFlow to allow dynamic GPU memory allocation for efficient computation.
### STEP 3:
Define Dataset Paths: Set directory paths for training and testing data to organize the data flow.
### STEP 4:
Load Sample Images: Visualize example images from both infected and uninfected classes to better understand the dataset.
### STEP 5:
Explore Data Dimensions: Check image dimensions and visualize distribution patterns to assess consistency.
### STEP 6:
Set Image Shape: Define a consistent image shape to standardize inputs for the model.
### STEP 7:
Data Augmentation: Use ImageDataGenerator to apply transformations like rotation, shift, zoom, and flipping to prevent overfitting.
### STEP 8:
Build Model Architecture: Create a CNN with convolutional and pooling layers to extract important features.
### STEP 9:
Add Dense Layers: Flatten the model and add dense layers for deeper learning, finishing with a sigmoid output layer.
### STEP 10:
Compile Model: Set the loss function, optimizer, and evaluation metrics to prepare the model for training.
### STEP 11:
Train the Model: Train the model over a specified number of epochs using the training and validation datasets.
### STEP 12:
Evaluate and Visualize: Assess model performance, generate metrics, and plot training and validation loss to monitor progress.

## PROGRAM

### Name: VASUNDRA SRI R

### Register Number: 212222230168
```
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.image import imread
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import utils
from tensorflow.keras import models
from sklearn.metrics import classification_report,confusion_matrix
import tensorflow as tf

# to share the GPU resources for multiple sessions
from tensorflow.compat.v1.keras.backend import set_session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True # dynamically grow the memory used on the GPU
config.log_device_placement = True # to log device placement (on which device the operation ran)
sess = tf.compat.v1.Session(config=config)
set_session(sess)

%matplotlib inline

my_data_dir = './dataset/cell_images'
os.listdir(my_data_dir)
test_path = my_data_dir+'/test/'
train_path = my_data_dir+'/train/'
os.listdir(train_path)
len(os.listdir(train_path+'/uninfected/'))
len(os.listdir(train_path+'/parasitized/'))
os.listdir(train_path+'/parasitized')[0]
para_img= imread(train_path+
                 '/parasitized/'+
                 os.listdir(train_path+'/parasitized')[0])
plt.imshow(para_img)

# Checking the image dimensions
dim1 = []
dim2 = []
for image_filename in os.listdir(test_path+'/uninfected'):
    img = imread(test_path+'/uninfected'+'/'+image_filename)
    d1,d2,colors = img.shape
    dim1.append(d1)
    dim2.append(d2)

sns.jointplot(x=dim1,y=dim2)
image_shape = (130,130,3)
help(ImageDataGenerator)
image_gen = ImageDataGenerator(rotation_range=20, # rotate the image 20 degrees
                               width_shift_range=0.10, # Shift the pic width by a max of 5%
                               height_shift_range=0.10, # Shift the pic height by a max of 5%
                               rescale=1/255, # Rescale the image by normalzing it.
                               shear_range=0.1, # Shear means cutting away part of the image (max 10%)
                               zoom_range=0.1, # Zoom in by 10% max
                               horizontal_flip=True, # Allo horizontal flipping
                               fill_mode='nearest' # Fill in missing pixels with the nearest filled value
                              )
image_gen.flow_from_directory(train_path)
image_gen.flow_from_directory(test_path)

model = models.Sequential()
model.add(keras.Input(shape=(image_shape)))
model.add(layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu',))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu',))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu',))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.summary()
batch_size = 16
train_image_gen = image_gen.flow_from_directory(train_path,target_size=image_shape[:2],
                                                color_mode='rgb',batch_size=batch_size,class_mode='binary')
test_image_gen = image_gen.flow_from_directory(test_path,target_size=image_shape[:2],
                                               color_mode='rgb',batch_size=batch_size,class_mode='binary',shuffle=False)
results = model.fit(train_image_gen,epochs=4,validation_data=test_image_gen)
model.save('cell_model.h5')
losses = pd.DataFrame(model.history.history)
print("VASUNDRA SRI R\n212222230168\n")
losses[['loss','val_loss']].plot()

model.metrics_names

import random
import tensorflow as tf
list_dir=["UnInfected","parasitized"]
dir_=(list_dir[1])
para_img= imread(train_path+ '/'+dir_+'/'+ os.listdir(train_path+'/'+dir_)[random.randint(0,100)])
img  = tf.convert_to_tensor(np.asarray(para_img))
img = tf.image.resize(img,(130,130))
img=img.numpy()
pred=bool(model.predict(img.reshape(1,130,130,3))<0.5 )
plt.title("Model prediction: "+("Parasitized" if pred
    else "Un Infected")+"\nActual Value: "+str(dir_))
plt.axis("off")
print("VASUNDRA SRI R\n212222230168\n")
plt.imshow(img)
plt.show()

model.evaluate(test_image_gen)
pred_probabilities = model.predict(test_image_gen)
print("VASUNDRA SRI R\n212222230168\n")
test_image_gen.classes
predictions = pred_probabilities > 0.5
print("VASUNDRA SRI R\n212222230168\n")
print(classification_report(test_image_gen.classes,predictions))
print("VASUNDRA SRI R\n212222230168\n")
confusion_matrix(test_image_gen.classes,predictions)
```

## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot
![Screenshot 2024-11-13 173745](https://github.com/user-attachments/assets/5c923dbb-a1bf-4f82-9ca7-d6b5161fd83c)

### Classification Report

![Screenshot 2024-11-13 174142](https://github.com/user-attachments/assets/ac20fd65-c972-4ca9-9fac-86676f61be0b)


### Confusion Matrix

![Screenshot 2024-11-13 174149](https://github.com/user-attachments/assets/dd1de234-47f5-4386-a35a-cb887f1bd193)


### New Sample Data Prediction
![Screenshot 2024-11-13 173712](https://github.com/user-attachments/assets/927f8ec7-160b-4cd0-82e2-67ec53ede351)

## RESULT
Thus, a deep neural network for Malaria infected cell recognition is developed and the performance is analyzed.
