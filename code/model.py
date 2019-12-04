import cv2
import numpy as np
from sampler import Sampler, Dataloader
import tensorflow as tf
from PIL import Image
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, Activation, MaxPooling2D, concatenate, add, Input, BatchNormalization, Activation, Dense, Dropout, Lambda
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Model, load_model
from tensorflow.python.keras import optimizers, losses



input_shape=[768, 768]





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

# train_sampler=Sampler(2, "train")

# a = model.fit_generator(train_sampler)




generator=Dataloader(1, "train")
a = model.fit_generator(
    generator.yielder(), epochs=150, steps_per_epoch=30, verbose=1)

# https://www.depends-on-the-definition.com/unet-keras-segmenting-images/