import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from keras.applications.efficientnet import  EfficientNetB1
from keras import callbacks
from keras.models import Sequential
import matplotlib.pyplot as plt
from datetime import datetime
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense
from keras import layers
from tensorflow.keras.optimizers import RMSprop, Adam
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES=True

IMG_WIDTH = 240
IMG_HEIGHT = 240

es_callback = callbacks.EarlyStopping(monitor='val_loss', patience=3,verbose=1,restore_best_weights=True)


efficient_net = EfficientNetB1(
    weights='imagenet',
    input_shape=(IMG_WIDTH,IMG_HEIGHT,3),
    include_top=False
)

model = Sequential()
model.add(efficient_net)
model.add(layers.GlobalAveragePooling2D())
#model.add(Dense(units = 512, activation = 'relu'))
model.add(layers.Dropout(0.2))
model.add(Dense(units = 1, activation='sigmoid'))
model.summary()

train_datagen = ImageDataGenerator(rescale=1./255,
		rotation_range=20,
		zoom_range=0.15,
		width_shift_range=0.2,
		height_shift_range=0.2,
		shear_range=0.15,
		horizontal_flip=True
        )

train_generator = train_datagen.flow_from_directory(
        './data/training/',  # This is the source directory for training images
        target_size=(IMG_WIDTH,IMG_HEIGHT),  # All images will be resized to 150x150
        batch_size=32,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')


validation_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = validation_datagen.flow_from_directory(
        './data/validation/',  # This is the source directory for training images
        target_size=(IMG_WIDTH,IMG_HEIGHT),  # All images will be resized to 150x150
        batch_size=32,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

opt1 = RMSprop(learning_rate=1e-5, decay=1e-5)
opt2 = Adam(learning_rate=1e-5) 

model.compile(optimizer=opt2, loss='binary_crossentropy', metrics=['acc'])

log_dir = 'logs/' + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


model.fit(
    train_generator,
    epochs = 110,
    steps_per_epoch = 8,
    validation_data = validation_generator,
    validation_steps = 8,
    callbacks=[tensorboard_callback])


test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
        './data/test/',  # This is the source directory for training images
        target_size=(IMG_WIDTH, IMG_HEIGHT),  # All images will be resized to 150x150
        batch_size=10,
        class_mode='binary')

image, label = test_generator.next()

for i in range(len(image)):
    my_image = np.expand_dims(image[i], 0)
    result = model.predict(my_image)
   
    print("\nPREDICCION: " + str(result))

    if result < 0.5:
            print("\nPREDICCION: Es Caballo:    valor:" + str(result))
    else:
            print("\nPREDICCION: Es Humano :    valor:" + str(result))#


    if label[i] == 0:
            print("REAL      : Es Caballo: ")
    else:
            print("REAL      : Es Humano : ")

    plt.imshow(image[i])
    plt.show()

model.save('saved-model/clasify_hoh')
