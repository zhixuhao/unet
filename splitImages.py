import skimage.io as io
import numpy as np
import os

def to_0_255_format_img(in_img):
    max_val = in_img[:,:].max()
    if max_val <= 1:
       out_img = np.round(in_img * 255)
       return out_img.astype(np.uint8)
    else:
        return in_img

def to_0_1_format_img(in_img):
    max_val = in_img[:,:].max()
    if max_val <= 1:
        return in_img
    else:
        out_img = in_img / 255
        return out_img

def split_image(img, tiled_name, save_dir = None, size = 256, overlap = 64, unique_area = 0):
    '''
    Split image to array of smaller images.
    
    Parameters
    ----------
    img : np.array
        Input image.
    tiled_name : list of np.arrays
        Tiled input imagr.
    save_dir : string, optional
        A folder with saved tiled images in png format. The default is "split_test/". If  save_dir = None, tiles don't save to hard drive.
    size : int, optional
        Size of tile side. Tiles are always square. The default is 256.
    overlap : int, optional
        Overlap by two tiles. The default is 64.
    unique_area : int, optional
        If resulting overlap of two tiles is too big, you can skip one tile. Used when preparing a training test. In test process should be 0. The default is 0.

    Returns
    -------
    1. Array of tiled images.
    2. Number of tiles per column, number of images per line.

    '''
    
    tiled_img = []
    h, w = img.shape[0:2]
    step = size - overlap
    count = 0
    rows = 0
    for y in range(0, h, step):
        rows += 1
        start_y = y
        end_y = start_y + size
        
        if (h - y <= size):
            if(h - y <= unique_area):
                break
            start_y = h - size
            end_y = h
        
        for x in range(0, w, step):
            start_x = x
            end_x = x + size

            if (w - x <= size):
                if (w - x < unique_area):
                    break
                start_x = w - size
                end_x = w
                
            tiled_img.append(img[start_y : end_y, start_x : end_x])
            if(save_dir != None):
                io.imsave( os.path.join(save_dir, (tiled_name + "_" + str(count) + ".png")), to_0_255_format_img(img[start_y : end_y, start_x : end_x]))
            count += 1
            if(end_x == w): # reached the end of the line
                break
        if(end_y == h):# reached the end of the height
            break
    
    cols = int(count / rows)
    return tiled_img, (rows, cols)

def glit_image(img_arr, out_size, tile_info, overlap = 64):
    '''
    Glit array of images to one big image

    Parameters
    ----------
    img_arr : list of np.array
        Tiles.
    out_size : (int, int)
        Shape of original image.
    tile_info : (int, int)
        Information about splitting (rows, cols).
    overlap : int, optional
        Overlap value. The default is 64.

    Returns
    -------
    np.array
        Glitted image.

    '''
    size = img_arr[0].shape[0]
    h, w = out_size[0:2]
    print(h,w)
    count_x = tile_info[1]
    count_y = tile_info[0]
    
    out = np.zeros(out_size)
    
    
    # corners
    out[0 : size, 0 : size] = img_arr[0]
    out[h - size : h, w - size : w] = img_arr[len(img_arr) - 1]
    out[0 : size, w - size : w] = img_arr[count_x - 1]
    out[h - size : h, 0 : size] = img_arr[len(img_arr) - count_x]
    
    half = int(overlap / 2)
    area = size - overlap
    
    for x in range(1, count_x - 1):
        #first row
        out[0 : size,     half + x * area : half + (x + 1) * area] = img_arr[x][0 : size, half : half + area] 
        #last row
        out[h - size : h, half + x * area : half + (x + 1) * area] = img_arr[(count_y - 1) * count_x + x][0 : size, half : size - half]

    for y in range(1, count_y - 1):
        # first column
        out[half  + y * area : half + (y + 1) * area, 0 : size] = img_arr[y * count_x][half : size - half, 0 : size] 
        # last column
        out[half  + y * area : half + (y + 1) * area, w - size : w] = img_arr[(y + 1) * count_x - 1][half : size - half, 0 : size] 
    
    
    # inner area
    for y in range(1, count_y - 1):
        for x in range(1, count_x - 1):
            out[half + y * area : half + (y + 1) * area, half + x * area : half + (x + 1) * area] = img_arr[y * count_x + x][half : size - half, half : size - half] 
    
    
    return to_0_255_format_img(out)

def test_split(filepath, filename, tiled_save_folder = "split_test", tiledfilename = "test"):
    
    if not os.path.isdir(tiled_save_folder):
        print("create output directory:" + tiled_save_folder)
        os.makedirs(tiled_save_folder)

    img = io.imread(os.path.join(filepath, filename), as_gray=True)
    img = to_0_1_format_img(img)
    arr, s = split_image(img, tiledfilename, save_dir = tiled_save_folder, size = 256, overlap = 64)

    print("x,y:", s)

    out = glit_image(arr, img.shape, s, overlap = 64)
    io.imsave(os.path.join(filepath, "test_out.png"), out)

    print("img-glit_out:", (to_0_255_format_img(img)-out).sum())

def test_glit(overlap=128,  glit_save_folder = "glit_test", glitfilename = "glit_test_white_black_square"): #white_black_square

        if (overlap == 128):  # (5,7) with overlap 128 and (4,5) with overlap 64
            count_x, count_y = (5, 7)
        elif (overlap == 64):
            count_x, count_y = (4, 5)
        else:
            print("no calculated data")
            return

        test_list = []
        for i in range(count_x * count_y):
            if i % 2 == 0:
                test_list.append(np.zeros((256, 256), np.float32))
            else:
                test_list.append(np.ones((256, 256), np.float32))

        res_img = glit_image(test_list, (768, 1024), (count_x, count_y), overlap=overlap)

        if not os.path.isdir(glit_save_folder):
            print("create out_dir:" + glit_save_folder)
            os.makedirs(glit_save_folder)
        io.imsave(os.path.join(glit_save_folder, glitfilename + ".png"), res_img)


if __name__ == "__main__":
    test_split('data\\test', "testing.png")
    test_glit()