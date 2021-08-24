from model import *
from data import *

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

num_class = 5

data_gen_args = dict(rotation_range=5,
                    width_shift_range=0.025,
                    height_shift_range=0.025,
                    shear_range=0.025,
                    zoom_range=0.025,
                    horizontal_flip=True,
                    fill_mode='nearest')

mask_name_label_list = ["mitochondria", "PSD", "vesicles", "axon", "boundaries", "mitochondrial boundaries"]

myGene = get_train_generator_data(dir_img_name = 'data/train/original',
                                  dir_mask_name = 'data/train/',
                                  aug_dict = data_gen_args,
                                  batch_size = 4,
                                  list_name_label_mask = mask_name_label_list,
                                  delete_mask_name = None,
                                  target_size = (256,256),
                                  color_mode_img = "gray",
                                  color_mode_mask = "gray",
                                  normalase_img_mod = "div255",
                                  num_class = num_class,
                                  label_mask = False,
                                  normalase_mask_mode="to_0_1", #"to_0_1"
                                  save_prefix_image="image_",
                                  save_prefix_mask="mask_",
                                  save_to_dir = None, #"data/myltidata/train4/temp",
                                  seed = 1,
                                  )

count = 0
for elem in myGene:
    x,y = elem
    print(x.shape, " ", y.shape)

    for i in range(x.shape[0]):
        #print(str("x")+":",  x[i].max(), " ",x[i].min())
        #print(str("y")+":",  y[i].max(), " ",y[i].min())

        cv2.imshow("test X"+str(i),  x[i])
        #for j in range(y.shape[-1]):
        #    cv2.imshow("test Y_" + mask_name_label_list[j],  y[i][:,:,j])
    cv2.waitKey()
    count+=1

