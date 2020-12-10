import copy
import json
import math
import os
import pickle

import cv2
import numpy as np

source_path = "your/path/to/directory/storing/images"
des_path = "your/path/to/directory/stroing/splitted/images"
json_path = "your/label/file"
new_json_path = "your/path/of/new/label/file"

with open(json_path) as json_file:
    data = json.load(json_file)
image_info =  []
image_info = data['images']

for img_info in image_info:
    img_path = os.path.join(source_path,img_info['file_name'])
    original_img_path = img_path
    img_name = os.path.basename(img_path)
    img_w_out_ext = os.path.splitext(img_name)[0]
    img_name_new = img_w_out_ext + "_000000{}{}".format(img_info['id'],".jpg")
    img_info['file_name'] = img_name_new
    img_info['path'] = des_path + "/" + img_name_new

    img = cv2.imread(original_img_path)
    cv2.imwrite(os.path.join(des_path,img_name_new),img)

data['images'] = image_info

ann_info = data['annotations']

for el in ann_info:
    el['num_keypoints'] = 17
    for i in range(11*3):
        el['keypoints'].append(0)

data['annotations'] = ann_info

with open(new_json_path, 'w') as f:
    json.dump(data, f)    


