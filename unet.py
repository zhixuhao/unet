import os 
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
from keras.models import *
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D, Concatenate
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from data import *

class myUnet(object):

	def __init__(self, img_rows = 512, img_cols = 512):

		self.img_rows = img_rows
		self.img_cols = img_cols

	def load_data(self):

		mydata = dataProcess(self.img_rows, self.img_cols)
		imgs_train, imgs_mask_train = mydata.load_train_data()
		imgs_test = mydata.load_test_data()
		return imgs_train, imgs_mask_train, imgs_test

	def get_unet(self):

	    inputs = Input((self.img_rows, self.img_cols,1))

		conv1 = Conv2D(64,
					3,
					activation='relu',
					padding='same',
					kernel_initializer='he_normal')(inputs)
		conv1 = Conv2D(64,
					3,
					activation='relu',
					padding='same',
					kernel_initializer='he_normal')(conv1)
		pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

		conv2 = Conv2D(128,
					3,
					activation='relu',
					padding='same',
					kernel_initializer='he_normal')(pool1)
		conv2 = Conv2D(128,
					3,
					activation='relu',
					padding='same',
					kernel_initializer='he_normal')(conv2)
		pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

		conv3 = Conv2D(256,
					3,
					activation='relu',
					padding='same',
					kernel_initializer='he_normal')(pool2)
		conv3 = Conv2D(256,
					3,
					activation='relu',
					padding='same',
					kernel_initializer='he_normal')(conv3)
		pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

		conv4 = Conv2D(512,
					3,
					activation='relu',
					padding='same',
					kernel_initializer='he_normal')(pool3)
		conv4 = Conv2D(512,
					3,
					activation='relu',
					padding='same',
					kernel_initializer='he_normal')(conv4)
		drop4 = Dropout(0.5)(conv4)
		pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

		conv5 = Conv2D(1024,
					3,
					activation='relu',
					padding='same',
					kernel_initializer='he_normal')(pool4)
		conv5 = Conv2D(1024,
					3,
					activation='relu',
					padding='same',
					kernel_initializer='he_normal')(conv5)
		drop5 = Dropout(0.5)(conv5)

		up6 = UpSampling2D(size=(2, 2))(drop5)
		up6 = Conv2D(512,
					2,
					activation='relu',
					padding='same',
					kernel_initializer='he_normal')(up6)
		merge6 = Concatenate(axis=3)([drop4, up6])
		conv6 = Conv2D(512,
					3,
					activation='relu',
					padding='same',
					kernel_initializer='he_normal')(merge6)
		conv6 = Conv2D(512,
					3,
					activation='relu',
					padding='same',
					kernel_initializer='he_normal')(conv6)

		up7 = UpSampling2D(size=(2, 2))(conv6)
		up7 = Conv2D(256,
					2,
					activation='relu',
					padding='same',
					kernel_initializer='he_normal')(up7)
		merge7 = Concatenate(axis=3)([conv3, up7])
		conv7 = Conv2D(256,
					3,
					activation='relu',
					padding='same',
					kernel_initializer='he_normal')(merge7)
		conv7 = Conv2D(256,
					3,
					activation='relu',
					padding='same',
					kernel_initializer='he_normal')(conv7)

		up8 = UpSampling2D(size=(2, 2))(conv7)
		up8 = Conv2D(128,
					2,
					activation='relu',
					padding='same',
					kernel_initializer='he_normal')(up8)
		merge8 = Concatenate(axis=3)([conv2, up8])
		conv8 = Conv2D(128,
					3,
					activation='relu',
					padding='same',
					kernel_initializer='he_normal')(merge8)
		conv8 = Conv2D(128,
					3,
					activation='relu',
					padding='same',
					kernel_initializer='he_normal')(conv8)

		up9 = UpSampling2D(size=(2, 2))(conv8)
		up9 = Conv2D(64,
					2,
					activation='relu',
					padding='same',
					kernel_initializer='he_normal')(up9)
		merge9 = Concatenate(axis=3)([conv1, up9])
		conv9 = Conv2D(64,
					3,
					activation='relu',
					padding='same',
					kernel_initializer='he_normal')(merge9)
		conv9 = Conv2D(64,
					3,
					activation='relu',
					padding='same',
					kernel_initializer='he_normal')(conv9)
		conv9 = Conv2D(2,
					3,
					activation='relu',
					padding='same',
					kernel_initializer='he_normal')(conv9)
		conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

		model = Model(inputs=inputs, outputs=conv10)
		model.compile(optimizer=Adam(lr=1e-4),
					loss='binary_crossentropy',
					metrics=['accuracy'])

		return model

	def train(self):

		print("loading data")
		imgs_train, imgs_mask_train, imgs_test = self.load_data()
		print("loading data done")
		model = self.get_unet()
		print("got unet")

		model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='loss',verbose=1, save_best_only=True)
		print('Fitting model...')
		model.fit(imgs_train, imgs_mask_train, batch_size=4, nb_epoch=10, verbose=1,validation_split=0.2, shuffle=True, callbacks=[model_checkpoint])

		print('predict test data')
		imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)
		np.save('../results/imgs_mask_test.npy', imgs_mask_test)

	def save_img(self):

		print("array to image")
		imgs = np.load('imgs_mask_test.npy')
		for i in range(imgs.shape[0]):
			img = imgs[i]
			img = array_to_img(img)
			img.save("../results/%d.jpg"%(i))




if __name__ == '__main__':
	myunet = myUnet()
	myunet.train()
	myunet.save_img()








