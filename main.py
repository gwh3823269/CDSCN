
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 18:36:50 2018
@author: Administrator
"""

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import merge
from keras.layers.convolutional import Convolution3D, AveragePooling2D,MaxPooling3D,Conv3D,Conv2D,Convolution2D
import keras.backend as K
from keras.optimizers import SGD, RMSprop,Adam
from keras.utils import np_utils, generic_utils
import scipy.io as sio
import tensorflow as tf
import numpy as np
from operator import truediv
import collections
from keras.layers.core import Reshape
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn import cross_validation
from keras.utils.np_utils import to_categorical
from keras.layers.normalization import  BatchNormalization
from keras.layers.merge import Concatenate
import keras
from keras.layers import add
from keras import layers
from keras.callbacks import EarlyStopping
import keras.callbacks as kcallbacks
from keras.regularizers import l2
from keras.layers import AveragePooling3D, Input
from keras.models import Model
from sklearn import metrics, preprocessing
uPavia = sio.loadmat('C:/Users/ADMIN/Desktop/SSRN-master/SSRN-master/datasets/UP/PaviaU.mat')
gt_uPavia = sio.loadmat('C:/Users/ADMIN/Desktop/SSRN-master/SSRN-master/datasets/UP/PaviaU_gt.mat')
data_IN = uPavia['paviaU']#高光谱数据，三维
gt_IN = gt_uPavia['paviaU_gt']#标签，二维
print (data_IN.shape)
def zeroPadding_3D(old_matrix, pad_length, pad_depth = 0):
    new_matrix = np.lib.pad(old_matrix, ((pad_length, pad_length), (pad_length, pad_length), (pad_depth, pad_depth)), 'constant', constant_values=0)
    return new_matrix

def indexToAssignment(index_, Row, Col, pad_length):
    new_assign = {}
    for counter, value in enumerate(index_):#enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中
        # enumerate多用于在for循环中得到计数，利用它可以同时获得索引和值，即需要index和value值的时候可以使用
        assign_0 = value // Col + pad_length#取整数
        assign_1 = value % Col + pad_length#取余数
        new_assign[counter] = [assign_0, assign_1]
    return new_assign

def assignmentToIndex( assign_0, assign_1, Row, Col):
    new_index = assign_0 * Col + assign_1
    return new_index
def selectNeighboringPatch(matrix, pos_row, pos_col, ex_len):
    selected_rows = matrix[range(pos_row-ex_len,pos_row+ex_len+1), :]
    selected_patch = selected_rows[:, range(pos_col-ex_len, pos_col+ex_len+1)]
    return selected_patch
def sampling(proptionVal, groundTruth):              #divide dataset into train and test datasets
    labels_loc = {}
    train = {}
    test = {}
    m = max(groundTruth)
    for i in range(m):
        indices = [j for j, x in enumerate(groundTruth.ravel().tolist()) if x == i + 1]
        np.random.shuffle(indices)
        labels_loc[i] = indices
        nb_val = int(proptionVal * len(indices))
        train[i] = indices[:-nb_val]
        test[i] = indices[-nb_val:]
#    whole_indices = []
    train_indices = []
    test_indices = []
    for i in range(m):
#        whole_indices += labels_loc[i]
        train_indices += train[i]
        test_indices += test[i]
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    return train_indices, test_indices
new_gt_IN = gt_IN

batch_size = 18
nb_classes = 9
nb_epoch = 200  #400
img_rows, img_cols = 7,7     #27, 27
patience = 200 #??

INPUT_DIMENSION_CONV = 103
INPUT_DIMENSION = 103

# 20%:10%:70% data for training, validation and testing

TOTAL_SIZE = 42776
VAL_SIZE = 4281
TRAIN_SIZE =4281 #20%

#TRAIN_SIZE =10698 #25%
TEST_SIZE = TOTAL_SIZE - TRAIN_SIZE
VALIDATION_SPLIT = 0.90                      # 20% for trainnig and 80% for validation and testing
# TRAIN_NUM = 10
# TRAIN_SIZE = TRAIN_NUM * nb_classes
# TEST_SIZE = TOTAL_SIZE - TRAIN_SIZE
# VAL_SIZE = TRAIN_SIZE

img_channels = 103
PATCH_LENGTH = 3               #Patch_size (13*2+1)*(13*2+1)

data = data_IN.reshape(np.prod(data_IN.shape[:2]),np.prod(data_IN.shape[2:]))#变形为（21025,200）的形式
gt = new_gt_IN.reshape(np.prod(new_gt_IN.shape[:2]),)#（21025，）
#np.prod()函数用来计算所有元素的乘积，对于有多个维度的数组可以指定轴，如axis=1指定计算每一行的乘积。
data = preprocessing.scale(data)#对数据进行标准化，标准化的过程为两步：去均值的中心化（均值变为0）；方差的规模化（方差变为1）

# scaler = preprocessing.MaxAbsScaler()
# data = scaler.fit_transform(data)

data_ = data.reshape(data_IN.shape[0], data_IN.shape[1],data_IN.shape[2])#将数据变形为（145,145,200）的形式

whole_data = data_
padded_data =zeroPadding_3D(whole_data, PATCH_LENGTH)#把数据（145,145,200）进行加边处理变形为（151,151,200）

ITER = 1
CATEGORY = 9 #16类

train_data = np.zeros((TRAIN_SIZE, 2*PATCH_LENGTH + 1, 2*PATCH_LENGTH + 1, INPUT_DIMENSION_CONV))#训练数据（2055,7,7,200）
val_data = np.zeros((VAL_SIZE, 2*PATCH_LENGTH + 1, 2*PATCH_LENGTH + 1, INPUT_DIMENSION_CONV))
test_data = np.zeros((TEST_SIZE, 2*PATCH_LENGTH + 1, 2*PATCH_LENGTH + 1, INPUT_DIMENSION_CONV))

seeds = [1334]
for index_iter in range(ITER):
    print("# %d Iteration" % (index_iter + 1))

    # save the best validated model 保证最佳的验证模型


    np.random.seed(seeds[index_iter])
#    train_indices, test_indices = sampleFixNum.samplingFixedNum(TRAIN_NUM, gt)
    train_indices, test_indices = sampling(VALIDATION_SPLIT, gt)

    # TRAIN_SIZE = len(train_indices)
    # print (TRAIN_SIZE)
    #
    # TEST_SIZE = TOTAL_SIZE - TRAIN_SIZE - VAL_SIZE
    # print (TEST_SIZE)

    y_train = gt[train_indices] - 1
    y_train = to_categorical(np.asarray(y_train))

    y_test = gt[test_indices] - 1
    y_test = to_categorical(np.asarray(y_test))

    # print ("Validation data:")
    # collections.Counter(y_test_raw[-VAL_SIZE:])
    # print ("Testing data:")
    # collections.Counter(y_test_raw[:-VAL_SIZE])

    train_assign = indexToAssignment(train_indices, whole_data.shape[0], whole_data.shape[1], PATCH_LENGTH)
    for i in range(len(train_assign)):
        train_data[i] = selectNeighboringPatch(padded_data, train_assign[i][0], train_assign[i][1], PATCH_LENGTH)

    test_assign = indexToAssignment(test_indices, whole_data.shape[0], whole_data.shape[1], PATCH_LENGTH)
    for i in range(len(test_assign)):
        test_data[i] = selectNeighboringPatch(padded_data, test_assign[i][0], test_assign[i][1], PATCH_LENGTH)

    x_train = train_data.reshape(train_data.shape[0], train_data.shape[1], train_data.shape[2], INPUT_DIMENSION_CONV)
    x_test_all = test_data.reshape(test_data.shape[0], test_data.shape[1], test_data.shape[2], INPUT_DIMENSION_CONV)

    x_val = x_test_all[-VAL_SIZE:]
    y_val = y_test[-VAL_SIZE:]

    x_test = x_test_all[:-VAL_SIZE]
    y_test = y_test[:-VAL_SIZE]
   
    print(x_train.shape, x_test.shape,x_val.shape)


x_train=x_train.reshape(-1,7,7,103,1)
x_test=x_test.reshape(-1,7,7,103,1)
x_val =x_val.reshape(-1,7,7,103,1)
input_shape=x_train.shape[1:]
n=2
depth = n * 6 + 2
version=1
model_type = 'ResNet%dv%d' % (depth, version)
def lr_schedule(nb_epoch):
    """Learning Rate Schedule学习速率表
    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.
    # Arguments
        epoch (int): The number of epochs
    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if nb_epoch > 180:
        lr *= 0.5e-3
    elif nb_epoch > 160:
        lr *= 1e-3
    elif nb_epoch > 120:
        lr *= 1e-2
    elif nb_epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr
  
def resnet_layer_first(inputs,
                 num_filters=24,
                 kernel_size=(1,1,7),
                 strides=(1,1,2),
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):#conv-bn-activation (True) or bn-activation-conv (False)
  

    conv = Conv3D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                   padding='valid',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x

def resnet_layer_first_1(inputs,
                 num_filters=24,
                 kernel_size=(1,1,7),
                 strides=(1,1,1),
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):#conv-bn-activation (True) or bn-activation-conv (False)
  

    conv = Conv3D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                   padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x
def resnet_layer_first_11(inputs,
                 num_filters=24,
                 kernel_size=(1,1,5),
                 strides=(1,1,1),
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):#conv-bn-activation (True) or bn-activation-conv (False)
  

    conv = Conv3D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                   padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x
def resnet_layer_second(inputs,
                 num_filters=24,
                 kernel_size=(3,3,1),
                 strides=(1,1,1),
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):#conv-bn-activation (True) or bn-activation-conv (False)
           
    conv = Conv3D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='valid',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x
def resnet_layer_second_1(inputs,
                 num_filters=24,
                 kernel_size=(3,3,1),
                 strides=(1,1,1),
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):#conv-bn-activation (True) or bn-activation-conv (False)
           
    conv = Conv3D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x
def resnet_layer_second_11(inputs,
                 num_filters=24,
                 kernel_size=(5,5,1),
                 strides=(1,1,1),
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):#conv-bn-activation (True) or bn-activation-conv (False)
           
    conv = Conv3D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x
def resnet_v1(input_shape, depth, num_classes=9):
    
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
   
    num_res_blocks = int((depth - 2) / 6)
    

    inputs = Input(shape=input_shape)
    # x=Conv3D(24,kernel_size=(7,1,1))
                 # strides=(2,1,1),
                 # padding='valid',
                 # kernel_initializer='he_normal',
                 # kernel_regularizer=l2(1e-4))
    x = resnet_layer_first(inputs=inputs)
    x1 = resnet_layer_first(inputs=inputs)
    # Instantiate the stack of residual units
    conv0_level1_shortcut = Convolution3D(24, (1, 1, 1), padding='same', strides=(1, 1,1),
                                          name='conv0_level1_shortcut')(x)
    for stack in range(1):
        for res_block in range(num_res_blocks):
            strides = (1,1,1)  # downsample
            y = resnet_layer_first_1(inputs=x,
                             num_filters=24,
                             strides=(1,1,1))
            y = resnet_layer_first_1(inputs=y,
                             num_filters=24,
                             strides=strides,
                             activation=None)
            y1 = resnet_layer_first_11(inputs=x1,
                             num_filters=24,
                             strides=(1,1,1))
            y1 = resnet_layer_first_11(inputs=y1,
                             num_filters=24,
                             strides=strides,
                             activation=None)
            c = merge([x , x1], mode='ave')
        x = keras.layers.add([c, y])
        x1 = keras.layers.add([c, y1])
    x = keras.layers.add([y, x1])       
    x = keras.layers.add([x, conv0_level1_shortcut])
    x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)
            #x = Activation('relu')(x)
    x = resnet_layer_first(inputs=x,
                                 num_filters=128,
                                 kernel_size=(1,1,49),
                                 strides=(1,1,1),
                                 activation=None,
                                 batch_normalization=False)#（7,7,1,128）
#变形
    y = Reshape((x._keras_shape[1],
                         x._keras_shape[2],
                         x._keras_shape[4],1))(x)#（7,7,128,1）

    x =resnet_layer_second(inputs=y,
                                 num_filters=24,
                                 kernel_size=(3,3,128),
                                 strides=(1,1,1),
                                 activation=None,
                                 batch_normalization=False)#(5,5,1,24)
    x1 =resnet_layer_second(inputs=y,
                                 num_filters=24,
                                 kernel_size=(3,3,128),
                                 strides=(1,1,1),
                                 activation=None,
                                 batch_normalization=False)#(5,5,1,24)
    conv1_level1_shortcut = Convolution3D(24, (1, 1,1), padding='same', strides=(1, 1,1),
                                          name='conv1_level1_shortcut')(x)#（7,7,97,24）
    for stack in range(1):
        for res_block in range(num_res_blocks):
            strides = (1,1,1)  # downsample
            y = resnet_layer_second_1(inputs=x,
                             num_filters=24,
                             strides=strides)
            y = resnet_layer_second_1(inputs=y,
                             num_filters=24,
                             activation=None)#(5,5,1,24)
            y1 = resnet_layer_second_11(inputs=x1,
                             num_filters=24,
                             strides=strides)
            y1= resnet_layer_second_11(inputs=y1,
                             num_filters=24,
                             activation=None)#(5,5,1,24)
            c1 = merge([x , x1], mode='mul')
        x = keras.layers.add([c1, y])
        x1 = keras.layers.add([c1, y1])
    x = keras.layers.add([y, x1]) 
    x = keras.layers.add([x, conv1_level1_shortcut])
    x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)
            #x = Activation('relu')(x)
    #x = AveragePooling3D(pool_size=(2,2,1))(x)
    x = AveragePooling3D(pool_size=(5,5,1))(x)
    
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)
    # Instantiate model.
    model= Model(inputs=inputs, outputs=outputs)
    return model    
model = resnet_v1(input_shape=input_shape, depth=depth)
model.summary()

#//////////

  



def get_categorical_accuracy_keras(y_ture,y_pred):
    return K.mean(K.equal(K.argmax(y_ture,axis=1),K.argmax(y_pred,axis=1)))   
# Compile
#model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=lr_schedule(nb_epoch)), metrics=[get_categorical_accuracy_keras])
model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=[get_categorical_accuracy_keras])
#model.fit(x_train,y_train,batch_size=16,epochs = 2,shuffle=True )
#用early_stopping返回最佳的epoch对应的model
#early_stopping=EarlyStopping(monitor='val_loss',patience=patience, verbose=1, mode='auto')
best_weights_ssrn_path_3d_2d = 'C:/Users/ADMIN/Desktop/SSRN(res)/models/UP-sum_9' + str(
        index_iter + 1) + '.hdf5'
#earlyStopping6 = kcallbacks.EarlyStopping(monitor='val_loss', patience=patience, verbose=1, mode='auto')
saveBestModel6 = kcallbacks.ModelCheckpoint(best_weights_ssrn_path_3d_2d, monitor='val_loss', verbose=1,
                                               save_best_only=True,
                                               mode='auto')
history=model.fit(x_train, y_train, batch_size=18, epochs=200,shuffle=True, validation_data=(x_val, y_val),callbacks=[saveBestModel6])
pred_test = model.predict(
        x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], x_test.shape[3], 1)).argmax(axis=1)
collections.Counter(pred_test)
gt_test = gt[test_indices] - 1
#计算全局精度
overall_acc = metrics.accuracy_score(pred_test, gt_test[:-VAL_SIZE])
confusion_matrix_res4 = metrics.confusion_matrix(pred_test, gt_test[:-VAL_SIZE])
#计算每一类的精度和平均精度

def AA_andEachClassAccuracy(confusion_matrix):
    counter = confusion_matrix.shape[0]
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc
each_acc, average_acc=AA_andEachClassAccuracy(confusion_matrix_res4)
KAPPA_RES = []
OA_RES = []
AA_RES = []

ELEMENT_ACC_RES = np.zeros((ITER, CATEGORY))

kappa = metrics.cohen_kappa_score(pred_test, gt_test[:-VAL_SIZE])
KAPPA_RES.append(kappa)
OA_RES.append(overall_acc)
AA_RES.append(average_acc)


ELEMENT_ACC_RES[index_iter, :] = each_acc
print(each_acc)
print(average_acc)
print(kappa)
print(model.evaluate(x_test,y_test,batch_size=18,verbose=2))

from pylab import *
import pandas as pd
mpl.rcParams['font.sans-serif'] = ['Times New Roman']
font_size =12 # 字体大小
#fig_size = (6,4) # 图表大小
# 更新字体大小
mpl.rcParams['font.size'] = font_size
# 更新图表大小
#mpl.rcParams['figure.figsize'] = fig_size

plt.plot(history.history['loss'],color='g')
plt.plot(history.history['val_loss'],color='r')
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train_loss","Val-loss"],loc="upper left")
plt.savefig("C:/Users/ADMIN/Desktop/123/UP_loss.png")
plt.show()

plt.plot(history.history['get_categorical_accuracy_keras'],color='g')
plt.plot(history.history['val_get_categorical_accuracy_keras'],color='r')
plt.title("model accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(["train_acc","Val-acc"],loc="downper right")
plt.savefig("C:/Users/ADMIN/Desktop/123/UP_acc.png")
plt.show()

'''#/////////////////////////////////////////////////////////////////////
#test
#prepare test data
for index_iter in range(10):
    print("# %d Iteration" % (index_iter + 1))

    # save the best validated model 保证最佳的验证模型


    np.random.shuffle([index_iter])
#    train_indices, test_indices = sampleFixNum.samplingFixedNum(TRAIN_NUM, gt)
    train_indices, test_indices = sampling(VALIDATION_SPLIT, gt)

    # TRAIN_SIZE = len(train_indices)
    # print (TRAIN_SIZE)
    #
    # TEST_SIZE = TOTAL_SIZE - TRAIN_SIZE - VAL_SIZE
    # print (TEST_SIZE)

    y_train = gt[train_indices] - 1
    y_train = to_categorical(np.asarray(y_train))

    y_test = gt[test_indices] - 1
    y_test = to_categorical(np.asarray(y_test))

    # print ("Validation data:")
    # collections.Counter(y_test_raw[-VAL_SIZE:])
    # print ("Testing data:")
    # collections.Counter(y_test_raw[:-VAL_SIZE])

    train_assign = indexToAssignment(train_indices, whole_data.shape[0], whole_data.shape[1], PATCH_LENGTH)
    for i in range(len(train_assign)):
        train_data[i] = selectNeighboringPatch(padded_data, train_assign[i][0], train_assign[i][1], PATCH_LENGTH)

    test_assign = indexToAssignment(test_indices, whole_data.shape[0], whole_data.shape[1], PATCH_LENGTH)
    for i in range(len(test_assign)):
        test_data[i] = selectNeighboringPatch(padded_data, test_assign[i][0], test_assign[i][1], PATCH_LENGTH)

    x_train = train_data.reshape(train_data.shape[0], train_data.shape[1], train_data.shape[2], INPUT_DIMENSION_CONV)
    x_test_all = test_data.reshape(test_data.shape[0], test_data.shape[1], test_data.shape[2], INPUT_DIMENSION_CONV)

    x_val = x_test_all[-VAL_SIZE:]
    y_val = y_test[-VAL_SIZE:]

    x_test = x_test_all[:-VAL_SIZE]
    y_test = y_test[:-VAL_SIZE]
   
    print(x_train.shape, x_test.shape,x_val.shape)

    x_test=x_test.reshape(-1,7,7,103,1)

    final_model = resnet_v1(input_shape=input_shape, depth=depth)

    final_model.load_weights('C:/Users/ADMIN/Desktop/SSRN(res)/models/UP-sum_91.hdf5')
    def get_categorical_accuracy_keras(y_ture,y_pred):
       return K.mean(K.equal(K.argmax(y_ture,axis=1),K.argmax(y_pred,axis=1)))   
# Compile
    final_model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=[get_categorical_accuracy_keras])

    pred_test = model.predict(
          x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], x_test.shape[3], 1)).argmax(axis=1)
    collections.Counter(pred_test)
    gt_test = gt[test_indices] - 1
#计算全局精度
    overall_acc_res4 = metrics.accuracy_score(pred_test, gt_test[:-VAL_SIZE])
    confusion_matrix_res4 = metrics.confusion_matrix(pred_test, gt_test[:-VAL_SIZE])
#计算每一类的精度和平均精度
    def AA_andEachClassAccuracy(confusion_matrix):
        counter = confusion_matrix.shape[0]
        list_diag = np.diag(confusion_matrix)
        list_raw_sum = np.sum(confusion_matrix, axis=1)
        each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
        average_acc = np.mean(each_acc)
        return each_acc, average_acc
    each_acc_res4, average_acc_res4 =AA_andEachClassAccuracy(confusion_matrix_res4)
    kappa = metrics.cohen_kappa_score(pred_test, gt_test[:-VAL_SIZE])
    print(each_acc_res4)
    print(average_acc_res4)
    print(kappa)
    print(model.evaluate(x_test,y_test,batch_size=18,verbose=1))

path1='C:/Users/ADMIN/Desktop/SSRN(res)/records/IN_train_SS_10.txt'
                             
path2='C:/Users/ADMIN/Desktop/SSRN(res)/records/IN_train_SS_element_10.txt'

f = open(path1, 'a')
sentence0 = 'KAPPAs, mean_KAPPA ± std_KAPPA for each iteration are:' + str(KAPPA_RES) + str(np.mean(KAPPA_RES)) + ' ± ' + str(np.std(KAPPA_RES)) + '\n'
f.write(sentence0)
sentence1 = 'OAs, mean_OA ± std_OA for each iteration are:' + str(OA_RES) + str(np.mean(OA_RES)) + ' ± ' + str(np.std(OA_RES)) + '\n'
f.write(sentence1)
sentence2 = 'AAs, mean_AA ± std_AA for each iteration are:' + str(AA_RES) + str(np.mean(AA_RES)) + ' ± ' + str(np.std(AA_RES)) + '\n'
f.write(sentence2)
element_mean = np.mean(ELEMENT_ACC_RES, axis=0)
element_std = np.std(ELEMENT_ACC_RES, axis=0)
sentence5 = "Mean of all elements in confusion matrix:" + str(np.mean(ELEMENT_ACC_RES, axis=0)) + '\n'
f.write(sentence5)
sentence6 = "Standard deviation of all elements in confusion matrix" + str(np.std(ELEMENT_ACC_RES, axis=0)) + '\n'
f.write(sentence6)
f.close()
print_matrix = np.zeros((CATEGORY), dtype=object)
for i in range(CATEGORY):
     print_matrix[i] = str(element_mean[i]) + " ± " + str(element_std[i])

     np.savetxt(path2, print_matrix.astype(str), fmt='%s', delimiter="\t",
               newline='\n')
'''
