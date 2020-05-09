

'''

    U-Net: Convolutional Networks for Biomedical Image Segmentation
    https://arxiv.org/abs/1505.04597

'''


import tensorflow.keras
from tensorflow.keras.layers import Dense, Conv2D, Conv2DTranspose,Cropping2D, concatenate
from tensorflow.keras.layers import BatchNormalization, Activation, Dropout
from tensorflow.keras.layers import Input, Flatten, MaxPooling2D
from tensorflow.keras.models import Model


def unet():
    input_layer = Input((572, 572, 1))

    # First Downsample block
    conv1 = Conv2D(64, 3, activation='relu', padding='valid', kernel_initializer='he_normal', name='dblock_1_c1')(input_layer)
    conv1 = Conv2D(64, 3, activation='relu', padding='valid', kernel_initializer='he_normal', name='dblock_1_c2')(conv1)
    conv1 = Dropout(0.25, name='dblock_1_d')(conv1)
    pool1 = MaxPooling2D((2), name='dblock_1_mp')(conv1)

    # Second Downsample Block
    conv2 = Conv2D(128, 3, activation='relu', padding='valid', kernel_initializer='he_normal', name='dblock_2_c1')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='valid', kernel_initializer='he_normal', name='dblock_2_c2')(conv2)
    conv2 = Dropout(0.5, name='dblock_2_d')(conv2)
    pool2 = MaxPooling2D((2), name='dblock_2_mp')(conv2)

    # # Third Downsample Block
    conv3 = Conv2D(256, 3, activation='relu', padding='valid', kernel_initializer='he_normal', name='dblock_3_c1')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='valid', kernel_initializer='he_normal', name='dblock_3_c2')(conv3)
    conv3 = Dropout(0.5, name='dblock_3_d')(conv3)
    pool3 = MaxPooling2D((2), name='dblock_3_mp')(conv3)

    # # Fourth Downsample Block
    conv4 = Conv2D(512, 3, activation='relu', padding='valid', kernel_initializer='he_normal', name='dblock_4_c1')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='valid', kernel_initializer='he_normal', name='dblock_4_c2')(conv4)
    conv4 = Dropout(0.5, name='dblock_4_d')(conv4)
    pool4 = MaxPooling2D((2), name='dblock_4_mp')(conv4)

    # Middle block
    convm = Conv2D(1024, 3, activation='relu', padding='valid', kernel_initializer='he_normal', name='block_m_c1')(pool4)
    convm = Conv2D(1024, 3, activation='relu', padding='valid', kernel_initializer='he_normal', name='block_m_c2')(convm)

    # Fourth Upsample Block
    deconv4 = Conv2DTranspose(512, 3, strides=(2, 2), activation='relu', padding='same', kernel_initializer='he_normal', name='ublock_4_deconv')(convm)
    conv4_crop = Cropping2D(4)(conv4)
    merge4 = concatenate([deconv4, conv4_crop])
    merge4 = Dropout(0.5, name='ublock_4_d') (merge4)
    uconv4 = Conv2D(512, 3, activation='relu', padding='valid', kernel_initializer='he_normal', name='ublock_4_c1')(merge4)
    uconv4 = Conv2D(512, 3, activation='relu', padding='valid', kernel_initializer='he_normal', name='ublock_4_c2')(uconv4)

    # # Third Upsample Block
    deconv3 = Conv2DTranspose(256, 3, strides=(2, 2), activation='relu', padding='same', kernel_initializer='he_normal', name='ublock_3_deconv')(uconv4)
    conv3_crop = Cropping2D(16)(conv3)
    merge3 = concatenate([deconv3, conv3_crop])
    merge3 = Dropout(0.5, name='ublock_3_d') (merge3)
    uconv3 = Conv2D(256, 3, activation='relu', padding='valid', kernel_initializer='he_normal', name='ublock_3_c1')(merge3)
    uconv3 = Conv2D(256, 3, activation='relu', padding='valid', kernel_initializer='he_normal', name='ublock_3_c2')(uconv3)

    # # Second Upsample Block
    deconv2 = Conv2DTranspose(128, 3, strides=(2, 2), activation='relu', padding='same', kernel_initializer='he_normal', name='ublock_2_deconv')(uconv3)
    conv2_crop = Cropping2D(40)(conv2)
    merge2 = concatenate([deconv2, conv2_crop])
    merge2 = Dropout(0.5, name='ublock_2_d') (merge2)
    uconv2 = Conv2D(128, 3, activation='relu', padding='valid', kernel_initializer='he_normal', name='ublock_2_c1')(merge2)
    uconv2 = Conv2D(128, 3, activation='relu', padding='valid', kernel_initializer='he_normal', name='ublock_2_c2')(uconv2)

    # First Upsample Block
    deconv1 = Conv2DTranspose(64, 3, strides=(2, 2), activation='relu', padding='same', kernel_initializer='he_normal', name='ublock_1_deconv')(uconv2)
    conv1_crop = Cropping2D(88) (conv1)
    merge1 = concatenate([deconv1, conv1_crop])
    merge1 = Dropout(0.5, name='ublock_1_d') (merge1)
    uconv1 = Conv2D(64, 3, activation='relu', padding='valid', kernel_initializer='he_normal', name='ublock_1_c1')(merge1)
    uconv1 = Conv2D(64, 3, activation='relu', padding='valid', kernel_initializer='he_normal', name='ublock_1_c2')(uconv1)

    output_layer = Conv2D(2, (1,1), padding="same", activation="sigmoid")(uconv1)
    model = Model(input_layer, output_layer)

    return model
