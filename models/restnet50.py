import tensorflow.keras
from tensorflow.keras.layers import Dense, Conv2D, Add
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.layers import AveragePooling2D, Input, Flatten, ZeroPadding2D, MaxPooling2D
from keras.initializers import glorot_uniform
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model




def identity_block(X, f, filters, stage, block):

  # Defining name 
  conv_name_base = 'res' + str(stage) + block + '_branch'
  bn_name_base = 'bn' + str(stage) + block + '_branch'

  # Retrieving filters
  F1, F2, F3 = filters

  # Coping the input and later add this to main path
  X_shortcut = X

  ###### MAIN PATH #######
  # First component of main path
  X = Conv2D(filters= F1, kernel_size= (1, 1), strides= (1, 1), padding= 'valid', name= conv_name_base + '2a', kernel_initializer= glorot_uniform(seed= 0)) (X)
  X = BatchNormalization(axis=3, name= bn_name_base + '2a') (X)
  X = Activation('relu') (X)

  # Second component of main path
  X = Conv2D(filters= F2, kernel_size= (f, f), strides= (1, 1), padding= 'same', name= conv_name_base + '2b', kernel_initializer= glorot_uniform(seed= 0)) (X)
  X = BatchNormalization(axis=3, name= bn_name_base + '2b') (X)
  X = Activation('relu') (X)

  # Third component of main path
  X = Conv2D(filters= F3, kernel_size= (1, 1), strides= (1,1), padding= 'valid', name= conv_name_base + '2c', kernel_initializer= glorot_uniform(seed=0)) (X)
  X = BatchNormalization(axis=3, name= bn_name_base + '2c') (X)

  # Final component: add shortcut to main path and pass it through relu activation
  X = Add() ([X, X_shortcut])
  X = Activation('relu') (X)

  return X


def conv_block(X, f, filters, stage, block, s = 2):

  # Defining name 
  conv_name_base = 'res' + str(stage) + block + '_branch'
  bn_name_base = 'bn' + str(stage) + block + '_branch'

  # Retrieving filters
  F1, F2, F3 = filters

  # Coping the input and later add this to main path
  X_shortcut = X

  ###### MAIN PATH ######
  # First component of main path
  X = Conv2D(filters= F1, kernel_size= (1, 1), strides= (s, s), name= conv_name_base + '2a', kernel_initializer= glorot_uniform(seed=0)) (X)
  X = BatchNormalization(axis = 3, name= bn_name_base + '2a') (X)
  X = Activation('relu') (X)

  # Second component of main path
  X = Conv2D(filters= F2, kernel_size= (f, f), strides= (1, 1), padding= 'same', name= conv_name_base + '2b', kernel_initializer=glorot_uniform(seed=0)) (X)
  X = BatchNormalization(axis=3, name= bn_name_base + '2b') (X)
  X = Activation('relu') (X)

  X = Conv2D(filters= F3, kernel_size= (1, 1), strides= (1, 1), padding= 'valid', name= conv_name_base + '2c', kernel_initializer= glorot_uniform(seed=0)) (X)
  X = BatchNormalization(axis=3, name= bn_name_base + '2c') (X)

  ##### SHORTCUT PATH ######
  X_shortcut = Conv2D(filters= F3, kernel_size= (1, 1), strides= (s, s), name= conv_name_base + '1', kernel_initializer=glorot_uniform(seed=0)) (X_shortcut)
  X_shortcut = BatchNormalization(axis= 3, name= bn_name_base + '1') (X_shortcut)

  # Final Step: Add shortcut value to main path, and pass it through a ReLU activation
  X = Add() ([X, X_shortcut])
  X = Activation('relu') (X)


  return X




def ResNet50(input_shape, classes):
  # Define the input layer
  X_input = Input(input_shape)

  # Zero-Padding
  X = ZeroPadding2D((3, 3))(X_input)

  # Stage 1 
  X = Conv2D(filters= 64, kernel_size= (7, 7), strides= (2, 2), kernel_initializer= glorot_uniform(seed=0)) (X)
  X = BatchNormalization(axis= 3, name= 'bn_conv1') (X)
  X = Activation('relu') (X)
  X = MaxPooling2D((3, 3), strides= (2, 2)) (X)

  # Stage 2
  X = conv_block(X, f= 3,  filters= [64, 64, 256], stage= 2, block= 'a', s= 1)
  X = identity_block(X, 3, [64, 64, 256], stage= 2, block= 'b')
  X = identity_block(X, 3, [64, 64, 256], stage= 2, block= 'c')

  # Stage 3
  X = conv_block(X, f = 3, filters = [128, 128, 512], stage = 3, block='a', s = 2)
  X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
  X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
  X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')

  # Stage 4
  X = conv_block(X, f = 3, filters = [256, 256, 1024], stage = 4, block='a', s = 2)
  X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
  X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
  X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
  X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
  X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

  # Stage 5
  X = conv_block(X, f = 3, filters = [512, 512, 2048], stage = 5, block='a', s = 2)
  X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
  X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

  # AVGPOOL.
  X = AveragePooling2D((2, 2), name='avg_pool')(X)

  # output layer
  X = Flatten()(X)
  X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)

  # Create model
  model = Model(inputs = X_input, outputs = X, name='ResNet50')

  return model

model = ResNet50(input_shape= (64, 64, 3), classes= 2)
model.summary()