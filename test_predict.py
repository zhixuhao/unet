from unet import *
from data import *

mydata = dataProcess(512,512)

imgs_test = mydata.load_test_data()

#imgs_train, imgs_mask = mydata.load_train_data()

myunet = myUnet()

model = myunet.get_unet()

model.load_weights('unet.hdf5')

imgs_mask_test = model.predict(imgs_test, batch_size = 1, verbose=1)

np.save('imgs_mask_test_meantrain.npy', imgs_mask_test)

#imgs_predict_train = model.predict(imgs_train[0:30], batch_size = 1, verbose=1)

#np.save('imgs_predict_train.npy', imgs_predict_train)