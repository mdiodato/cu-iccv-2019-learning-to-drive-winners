import scipy
from scipy import ndimage
import pandas as pd
import os
import datetime

def list_files(dir):
    r = []
    for root, dirs, files in os.walk(dir):
        for name in files:
            r.append(os.path.join(root, name))
    return r

def resize_image(f_name, output_name, resize_shape=(320, 180)):
    image = ndimage.imread(f_name, flatten=False)
    sampled_image = scipy.misc.imresize(image, size=resize_shape)
    scipy.misc.imsave(output_name, sampled_image)
    


input_folder = './Drive360Images/'
output_folder = './Drive360Images_320_180/'

files = pd.concat([pd.read_csv('test_full.csv'), pd.read_csv('train_full.csv'), pd.read_csv('val_full.csv')])
cols = 4

for i in range(len(files)):
    if i % 1000 == 0:
        print(i, i/(cols*len(files)), datetime.datetime.now())
    for j in range(cols):
        file_in = main_dir + input_folder + files.iloc[i,j]
        if file_in.endswith('jpg'):
            file_out = main_dir + output_folder + files.iloc[i,j]
            if not os.path.exists(os.path.dirname(file_out)):
                os.makedirs(os.path.dirname(file_out))
            if not os.path.exists(file_out):
                resize_image(file_in, file_out, (320, 180))

