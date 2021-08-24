from keras.models import *
from keras.layers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
import keras

from keras.utils.vis_utils import plot_model
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

def unet(pretrained_weights = None,input_size = (256,256,1), num_class = 2):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    normal1 = (BatchNormalization())(conv1)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(normal1)
    normal1 = (BatchNormalization())(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(normal1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    normal2 = (BatchNormalization())(conv2)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(normal2)
    normal2 = (BatchNormalization())(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(normal2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    normal3 = (BatchNormalization())(conv3)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(normal3)
    normal3 = (BatchNormalization())(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(normal3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    normal4 = (BatchNormalization())(conv4)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(normal4)
    normal4 = (BatchNormalization())(conv4)
    drop4 = Dropout(0.5)(normal4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    normal5 = (BatchNormalization())(conv5)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(normal5)
    normal5 = (BatchNormalization())(conv5)
    drop5 = Dropout(0.5)(normal5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    normal6 = (BatchNormalization())(conv6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(normal6)
    normal6 = (BatchNormalization())(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(normal6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    normal7 = (BatchNormalization())(conv7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(normal7)
    normal7 = (BatchNormalization())(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(normal7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    normal8 = (BatchNormalization())(conv8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(normal8)
    normal8 = (BatchNormalization())(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(normal8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    normal9 = (BatchNormalization())(conv9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(normal9)
    normal9 = (BatchNormalization())(conv9)

    dense10 = Dense(num_class)(normal9)
    conv10 = Activation('sigmoid')(dense10)

    model = Model(inputs=inputs, outputs=conv10)

    #model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['categorical_accuracy'])

    if num_class == 1:
        model.compile(optimizer='adam',
                      loss=[dice_coef_loss],
                      metrics=[dice_coef])
    elif num_class == 5:
        model.compile(optimizer='rmsprop',
                      loss=[dice_coef_loss_multilabel5],
                      # loss=['binary_crossentropy'],
                      # loss=['categorical_crossentropy'],
                      metrics=[dice_coef_multilabel5]
                      #,loss_weights = [0.1,0.1,0.1,1.0,0.1]
                      )
    elif num_class == 6:
        model.compile(optimizer='rmsprop',
                      loss=[dice_coef_loss_multilabel6],
                      # loss=['binary_crossentropy'],
                      # loss=['categorical_crossentropy'],
                      metrics=[dice_coef_multilabel6]
                      #,loss_weights = [0.1,0.1,0.1,1.0,0.1]
                      )
    #print(model.summary())
    #plot_model(model, show_shapes=True, show_layer_names=True, expand_nested=True, to_file="model1.png")

    #plot_model(model, show_shapes=True, show_layer_names=True, expand_nested=False, to_file="model2.png")
    #plot_model(model, show_shapes=True, show_layer_names=False, expand_nested=True, to_file="model3.png")
    #plot_model(model, show_shapes=False, show_layer_names=True, expand_nested=True, to_file="model4.png")

    #plot_model(model, show_shapes=False, show_layer_names=False, expand_nested=False, to_file="model5.png")

    #plot_model(model, show_shapes=False, show_layer_names=False, expand_nested=True, to_file="model6.png")
    #plot_model(model, show_shapes=False, show_layer_names=True, expand_nested=False, to_file="model7.png")
    #plot_model(model, show_shapes=True, show_layer_names=False, expand_nested=False, to_file="model8.png")



    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

from keras import backend as K

def dice_coef(y_true, y_pred):
    smooth = 0.0001
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)


#numLabels = 5 or 6

def dice_coef_multilabel2(y_true, y_pred, numLabels = 2):
    dice=0
    for index in range(numLabels):
        dice += dice_coef(y_true[:,:,:,index], y_pred[:,:,:,index])
    return dice/numLabels # taking average


def dice_coef_loss_multilabel2(y_true, y_pred, numLabels = 2):
    return 1-dice_coef_multilabel2(y_true, y_pred, numLabels)


def dice_coef_multilabel5(y_true, y_pred, numLabels = 5):
    dice=0
    for index in range(numLabels):
        dice += dice_coef(y_true[:,:,:,index], y_pred[:,:,:,index])
    return dice/numLabels # taking average


def dice_coef_loss_multilabel5(y_true, y_pred, numLabels = 5):
    return 1-dice_coef_multilabel5(y_true, y_pred, numLabels)


def dice_coef_multilabel6(y_true, y_pred, numLabels = 6):
    dice=0
    for index in range(numLabels):
        dice += dice_coef(y_true[:,:,:,index], y_pred[:,:,:,index])
    return dice/numLabels # taking average


def dice_coef_loss_multilabel6(y_true, y_pred, numLabels = 6):
    return 1-dice_coef_multilabel6(y_true, y_pred, numLabels)