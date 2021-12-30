from model import *
from data import *
import matplotlib.pyplot as plt

num_class = 6

data_gen_args = dict(rotation_range= 5,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    vertical_flip=True,
                    fill_mode='nearest')

#берутся первые классы из списка
mask_name_label_list = ["mitochondrion", "PSD", "vesicles", "axon", "membranes", "mitochondrial boundaries"]

myGene = get_train_generator_data(dir_img_name = 'data/epfl_train/slices',
                                  dir_mask_name = 'data/epfl_train/',
                                  aug_dict = data_gen_args,
                                  batch_size = 1,
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

model_checkpoint = ModelCheckpoint('my_unet_multidata_pe69_bs9_6class.hdf5', mode='auto', monitor='loss',verbose=1, save_best_only=True)

history = model.fit(myGene, steps_per_epoch=15, epochs=2, callbacks=[model_checkpoint], verbose=1, validation_data=myGene, validation_steps=5)

#save history
import json
with open('training_history_pe69_bs9_6class.json', 'w') as file:
    json.dump(history.history, file, indent=4)
# Обучение и проверка точности значений

plt.plot(history.history["dice_coef_multilabel"])
plt.plot(history.history["val_dice_coef_multilabel"])
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