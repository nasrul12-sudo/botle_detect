import os
import random
import shutil
import numpy as np
import pathlib
import tensorflow as tf 
import matplotlib.pyplot as plt

from PIL import Image
from skimage import io
from keras_preprocessing.image import load_img
from keras_preprocessing.image import img_to_array
from keras_preprocessing.image import ImageDataGenerator
from keras.layers.core import Dense
from keras.layers.core import Flatten
from keras import layers
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import BatchNormalization

def dir(dir):
    labels = os.listdir(dir)
    output_path = 'water_table_datasets'

    if not os.path.exists(output_path):
        # print(output_path)
        os.mkdir(os.path.join(output_path))    

    if not os.path.exists(output_path):
        os.mkdir(os.path.join(output_path))

    for label in labels:
        sub_label = os.path.join(dir,label)
        sub_labels_path = os.path.join(sub_label, label)
        label_path = os.listdir(sub_labels_path)
        ims = [i for i in os.listdir(sub_labels_path) if i.endswith(".jpeg")]
        random.shuffle(ims)
        split_size = 0.8
        train_len = int(len(ims) * split_size)
        train_ims = ims[:train_len]
        val_ims = ims[train_len:]

        # create train and val dirs
        train_path = os.path.join(output_path, "train")
        label_train_path = os.path.join(train_path, label)


        val_path = os.path.join(output_path, "val")
        label_val_path = os.path.join(val_path, label)

        if not os.path.exists(train_path):
            os.mkdir(train_path)

        if not os.path.exists(label_train_path):
            os.mkdir(label_train_path)

        if not os.path.exists(val_path):
            os.mkdir(val_path)  

        if not os.path.exists(label_val_path):
            os.mkdir(label_val_path)

        for im in train_ims:
            shutil.copy(os.path.join(sub_labels_path, im), label_train_path)

        for im in val_ims:
            shutil.copy(os.path.join(sub_labels_path, im), label_val_path)

    return labels, train_path, val_path

def training(labels, train_path, val_path):
    datagen = ImageDataGenerator(        
                width_shift_range=0.1,  
                height_shift_range=0.1,    
                brightness_range = (0.3, 0.9),
                zoom_range=0.2)


    for label in labels:
        image_directory = train_path + '/' + label + '/'
        SIZE = 150
        dataset = []

        print(image_directory)
        my_images = os.listdir(image_directory)
        for i, image_name in enumerate(my_images):    
            if ((image_name.split('.')[1] == 'jpeg')):
                image = load_img(image_directory + image_name, target_size = (150,150))
                image = img_to_array(image)
                dataset.append(image)

        x = np.array(dataset)
        i = 0
        for _ in datagen.flow(x, batch_size=16,
                                save_to_dir= train_path + '/' + label + '/',
                                save_prefix='aug',
                                save_format='jpeg'):
            i += 1    
            if i > 50:        
                break

        

        for label in labels:
            image_directory = val_path + '/' + label + '/'
            SIZE = 150
            dataset = []

            print(image_directory)
            my_images = os.listdir(image_directory)
            for i, image_name in enumerate(my_images):    
                if ((image_name.split('.')[1] == 'jpeg')):
                    image = load_img(image_directory + image_name, target_size = (150,150))
                    image = img_to_array(image)
                    dataset.append(image)

            x = np.array(dataset)
            i = 0
            for _ in datagen.flow(x, batch_size=16,
                                    save_to_dir= val_path + '/' + label + '/',
                                    save_prefix='aug',
                                    save_format='jpeg'):
                i += 1    
                if i > 20:        
                    break

def Models(train_path, val_path):
    train_dataset_url = train_path
    train_data_dir = pathlib.Path(train_dataset_url)

    validation_dataset_url = val_path
    validation_data_dir = pathlib.Path(validation_dataset_url)

    img_height,img_width= 150,150
    batch_size=16
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
      train_data_dir,
      seed=123,
      image_size=(img_height, img_width),
      batch_size=batch_size)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    validation_data_dir,
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
        for i in range(6):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.axis("off")

    model = Sequential([
        layers.Conv2D(32, (5,5), activation='relu', padding= 'valid', input_shape=(150,150,3)),
        layers.MaxPooling2D(pool_size=(2,2)),
        tf.keras.layers.Dropout(0.2),
        BatchNormalization(),

        layers.Conv2D(32, (5,5), activation='relu'),
        layers.MaxPooling2D(pool_size=(2,2)),
        tf.keras.layers.Dropout(0.2),
        BatchNormalization(),

        layers.Conv2D(32, (3,3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2,2)),
        tf.keras.layers.Dropout(0.2),
        BatchNormalization(),

        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2,2)),
        tf.keras.layers.Dropout(0.2),
        BatchNormalization(),

        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2,2)),

        Flatten(),
        Dense(512, activation='relu'),
        Dense(3, activation='softmax'),
    ])

    model.compile(optimizer=Adam(lr=0.0001),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    model.fit_generator(train_ds, validation_data=val_ds, epochs=30)
    model.save("botle.h5")


datasets = 'temp/'
data = os.listdir(datasets)

Dir = dir(datasets)
label = Dir[0]
training_path = Dir[1]
validation_path = Dir[2]

# training(label, training_path, validation_path)
Models(training_path, validation_path)