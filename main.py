from model import *
from data import *
import matplotlib.pyplot as plt
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#print(tf.test.is_built_with_cuda())
#print(tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None))

#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
#print("GPUs: ", len(tf.config.experimental.list_physical_devices('GPU')))

#from tensorflow.compat.v1 import ConfigProto
#from tensorflow.compat.v1 import InteractiveSession

#config = ConfigProto()
#config.gpu_options.allow_growth = True
#session = InteractiveSession(config=config)

#GPU desable
try:
    # Disable all GPUS
   tf.config.set_visible_devices([], 'GPU')
   visible_devices = tf.config.get_visible_devices()
   for device in visible_devices:
       assert device.device_type != 'GPU'
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

num_class = 1

data_gen_args = dict(rotation_range= 5,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    vertical_flip=True,
                    fill_mode='nearest')

#берутся первые классы из списка
mask_name_label_list = ["mitochondria", "PSD", "vesicles", "axon", "boundaries", "mitochondrial boundaries"]

myGene = get_train_generator_data(dir_img_name = 'data/train/original',
                                  dir_mask_name = 'data/train/',
                                  aug_dict = data_gen_args,
                                  batch_size = 9,
                                  list_name_label_mask = mask_name_label_list,
                                  delete_mask_name = None,
                                  target_size = (256,256),
                                  color_mode_img = "gray",
                                  color_mode_mask = "gray",
                                  normalase_img_mod = "div255",
                                  num_class = num_class,
                                  label_mask = False,
                                  normalase_mask_mode = "to_0_1", #"to_-1_1"
                                  save_prefix_image="image_",
                                  save_prefix_mask="mask_",
                                  save_to_dir = None, #"/content/drive/MyDrive/Calab/data/myltidata/train4/temp",
                                  seed = 1
                                  )

model = unet(num_class = num_class)
#model = unet('my_unet_multidata_pe69_bs9_1class.hdf5', num_class = num_class)

model_checkpoint = ModelCheckpoint('my_unet_multidata_pe69_bs9_1class.hdf5', mode='auto', monitor='loss',verbose=1, save_best_only=True)

history = model.fit(myGene, steps_per_epoch=69, epochs=100, callbacks=[model_checkpoint], verbose=1, validation_data=myGene, validation_steps=17)

#save history
import json
with open('training_history_pe69_bs9_1class.json', 'w') as file:
    json.dump(history.history, file, indent=4)
# Обучение и проверка точности значений

plt.plot(history.history["dice_coef"])
plt.plot(history.history["val_dice_coef"])
plt.title("Model Dice")
plt.ylabel("Dice")
plt.xlabel("Epoch")
plt.legend(["Train", "Validation"], loc="upper left")
plt.show()

# Обучение и проверка величины потерь

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Model loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["Train", "Validation"], loc="upper left")
plt.show()