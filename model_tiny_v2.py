from keras.models import *
from keras.layers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
import tensorflow as tf
from metrics import *

def unet(pretrained_weights=None, input_size=(256, 256, 1), num_class=2):
    inputs = Input(input_size)
    conv1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    normal1 = (BatchNormalization())(conv1)
    conv1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(normal1)
    normal1 = (BatchNormalization())(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(normal1)

    conv2 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    normal2 = (BatchNormalization())(conv2)
    conv2 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(normal2)
    normal2 = (BatchNormalization())(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(normal2)

    conv3 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    normal3 = (BatchNormalization())(conv3)
    conv3 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(normal3)
    normal3 = (BatchNormalization())(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(normal3)

    conv4 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    normal4 = (BatchNormalization())(conv4)
    conv4 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(normal4)
    normal4 = (BatchNormalization())(conv4)
    drop4 = Dropout(0.5)(normal4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    normal5 = (BatchNormalization())(conv5)
    conv5 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(normal5)
    normal5 = (BatchNormalization())(conv5)
    drop5 = Dropout(0.5)(normal5)

    up6 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    normal6 = (BatchNormalization())(conv6)
    conv6 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(normal6)
    normal6 = (BatchNormalization())(conv6)

    up7 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(normal6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    normal7 = (BatchNormalization())(conv7)
    conv7 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(normal7)
    normal7 = (BatchNormalization())(conv7)

    up8 = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(normal7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    normal8 = (BatchNormalization())(conv8)
    conv8 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(normal8)
    normal8 = (BatchNormalization())(conv8)

    up9 = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(normal8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    normal9 = (BatchNormalization())(conv9)
    conv9 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(normal9)
    normal9 = (BatchNormalization())(conv9)
    
    dense10 = Dense(num_class, activation='sigmoid')(normal9)

    model = Model(inputs=inputs, outputs=dense10)

    # model.compile(optimizer = Adam(learning_rate = 1e-4), loss = 'binary_crossentropy', metrics = ['categorical_accuracy'])

    model.compile(optimizer=tf.optimizers.Adam(learning_rate = 5e-3),
                  loss=[universal_dice_coef_loss(num_class)],
                  metrics=[universal_dice_coef_multilabel(num_class)])

    # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model
    
if __name__ == "__main__":
    model = unet()
    model.summary()
