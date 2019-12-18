import cv2
import os
import pickle
import numpy as np
from sampler import Sampler, test_generator
import tensorflow as tf
from PIL import Image
from random import shuffle
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, Activation, MaxPooling2D, concatenate, add, Input, BatchNormalization, Activation, Dense, Dropout, Lambda
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Model, load_model
from tensorflow.python.keras import optimizers, losses
from utils import create_list


# #Just use part of GPU memory:
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))




input_shape=[768, 768]
data_partitions=[0.8,0.2] #The portion of training and valiating data. If data_partitions[0]=0.8 then 80% od the data is training data




def conv_block(input_tensor, n_filters, batchnorm, kernel_size=3):
    #1st layer
    x= Conv2D(n_filters, kernel_size=(kernel_size,kernel_size), padding="same")(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)

    # second layer
    
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x




def model(input_shape, n_filters=16, batchnorm=False, dropout=False):

    inputs = Input(shape=(input_shape[1], input_shape[0],3))

    x = Lambda(lambda i: (tf.to_float(i) / 255))(inputs)
    #down

    c1 = conv_block(x, n_filters, batchnorm)
    pool1=MaxPooling2D(pool_size=(2,2), strides=None)(c1)

    c2 = conv_block(pool1, n_filters*2, batchnorm)
    pool2=MaxPooling2D(pool_size=(2,2), strides=None)(c2)

    c3 = conv_block(pool2, n_filters*4, batchnorm)
    pool3=MaxPooling2D(pool_size=(2,2), strides=None)(c3)

    c4 = conv_block(pool3, n_filters*8, batchnorm)
    #up


    u1=Conv2DTranspose(n_filters*4, kernel_size=(3,3), strides=(2, 2), padding='same')(c4)
    u1=concatenate([u1, c3])
    u1=conv_block(u1, n_filters*4, batchnorm)

    u2=Conv2DTranspose(n_filters*2, kernel_size=(3,3), strides=(2, 2), padding='same')(u1)
    u2=concatenate([u2, c2])
    u2=conv_block(u2, n_filters*2, batchnorm)

    u3=Conv2DTranspose(n_filters, kernel_size=(3,3), strides=(2, 2), padding='same') (u2)
    u3=concatenate([u3, c1])
    u3=conv_block(u3, n_filters, batchnorm)



    outputs = Conv2D(3, (1, 1), activation='sigmoid') (u3)
    model = Model(inputs=inputs, outputs=outputs)
    return model



optimizer = optimizers.SGD(
        lr=0.0005, decay=1e-6, momentum=0.9, nesterov=True
        )






model=model(input_shape)
model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=['accuracy'])
model.summary()


#Creating the Test and Validation data lists
if not os.path.exists("/home/ubuntu/martin/kaggle/data/data_list.pkl"):
    data=create_list()
    output = open("/home/ubuntu/martin/kaggle/data/data_list.pkl", 'wb')
    pickle.dump(data, output)
    output.close()

else:
    pkl_file = open("/home/ubuntu/martin/kaggle/data/data_list.pkl", 'rb')
    data = pickle.load(pkl_file)
    pkl_file.close()


shuffle(data) #This line is just to not have the same training data every time
train_border=int(len(data)*data_partitions[0])
print(train_border)
import pdb
pdb.set_trace()
train_data=data[:train_border]
valid_data=data[train_border:]



early_stopping=EarlyStopping(monitor="acc", patience=15, verbose=1, mode="max")
checkpoint_path="/home/ubuntu/martin/kaggle/ship-detection-on-airbus-photos/checkpoint.hdf5"
checkpoint = ModelCheckpoint(
    checkpoint_path,
    period=1,
    monitor="acc",
    verbose=1,
    save_best_only=True,
    mode="max",
)


#Initiating the samplers for the fit_generator
train_sampler=Sampler(batch_size=10, data=train_data, data_type="train")
valid_sampler=Sampler(batch_size=10, data=valid_data, data_type="valid")
a = model.fit_generator(train_sampler, epochs=150, verbose=1, validation_data=valid_sampler, max_queue_size=100,
    workers=32, use_multiprocessing=True, callbacks=[early_stopping, checkpoint])



model.load_weights(checkpoint_path)

root="/home/ubuntu/martin/kaggle/data/test_data"
test_list=["75f89af6f.jpg", "75f1022b4.jpg", "75f4396e7.jpg", "75f4880cd.jpg", "75f6217a4.jpg", "75f9225f6.jpg", "075f59258.jpg", "75f140408.jpg", "75f419101.jpg", "75f771352.jpg"]
for i in range (len(test_list)):
    test_list[i]=root+test_list[i]
test_generator(test_list)
images=model.predict(test_generator, batch_size=1, verbose=0, steps=len(test_list))


import pdb
pdb.set_trace()

for i in range (len(images)):
    cv2.imsave(images[i], "/home/ubuntu/martin/kaggle/result_{}".format(test_list[i][-14:]))

# https://www.depends-on-the-definition.com/unet-keras-segmenting-images/