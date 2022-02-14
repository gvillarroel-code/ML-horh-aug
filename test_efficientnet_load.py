import numpy as np # linear algebra
import os, os.path
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from keras.applications.efficientnet import  EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3,EfficientNetB4, EfficientNetB5, EfficientNetB7
from keras.models import Sequential
import matplotlib.pyplot as plt
from datetime import datetime
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense
from keras import layers, models, applications, callbacks
from tensorflow.keras.optimizers import RMSprop, Adam

IMG_WIDTH = 240
IMG_HEIGHT = 240

#########################################
#      LOAD PREVIUSLY SAVED MODEL       # 
#########################################
loaded_model = models.load_model('saved-model/clasify_hoh')

#########################################
#    COUNT FILES IN TEST DIRECTORY      #
#########################################
nfiles=0
for files in os.walk(r'./data/test'):
        nfiles += len(files[2])
        nfiles -= 1

#########################################
#    LOAD IMAGES FROM TEST DIRECTORY    #
#########################################
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
        './data/test/',  # This is the source directory for training images
        target_size=(IMG_WIDTH, IMG_HEIGHT),  # All images will be resized to 150x150
        batch_size=30,
        class_mode='binary')


#########################################
#        SHOW AND PREDICT IMAGES        #
#########################################
image, label = test_generator.next()

for i in range(len(image)):
    my_image = np.expand_dims(image[i], 0)
    result = loaded_model.predict(my_image)
   
    print("--------------\n")

    if result < 0.5:
            print("\nPREDICCION: Es Caballo")
    else:
            print("\nPREDICCION: Es Humano")


    if label[i] == 0:
            print("REAL      : Es Caballo")
    else:
            print("REAL      : Es Humano")

    plt.imshow(image[i])
    plt.show()
