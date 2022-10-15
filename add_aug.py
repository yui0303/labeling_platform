import xml.etree.ElementTree as ET
import os
import shutil
import json
import numpy as np
from PIL import Image

def salt_and_pepper(image, prob=0.05):
    arr = np.asarray(image)
    choice = np.random.choice(3, arr.shape[:2], p = [prob/2, 1 - prob, prob/2]) # 0: black, 1:original , 2: white
    
    for i in range(3):
        arr[...,i] =  np.where(choice == 0, 0, arr[...,i])
        arr[...,i] =  np.where(choice == 2, 255, arr[...,i])
    return Image.fromarray(arr)



with open("mAP/ap.json", 'r') as apf:
    data = json.load(apf)
if os.path.exists('augmentation'): 
    for file in os.listdir('augmentation'):
        os.remove('./augmentation/' + file)
else: os.mkdir('augmentation')
target_dir = './prepare_data/with_annotation/train/'
dest_dir = './augmentation/'
xml_list = [x for x in os.listdir(target_dir) if x.endswith('xml')]
minObjItem = min(data, key=lambda x:x['AP'])
minObj = minObjItem["Obj"]
for file in xml_list:
    filename_without_extension = file.split('.')[0]
    tree = ET.parse(os.path.join(target_dir,file))
    root = tree.getroot()
    for obj in root.findall('object'):
        root.find('filename').text = filename_without_extension + '_noise.jpg'
        if obj.find('name').text == minObj:
            tree.write(os.path.join(dest_dir, filename_without_extension + '_noise.xml'))
            #shutil.copy(os.path.join(target_dir,filename_without_extension + '.jpg'), os.path.join(dest_dir, filename_without_extension + '_1.jpg')) # copy jpg to argumentation dir
            img = Image.open(os.path.join(target_dir,filename_without_extension + '.jpg')).convert('RGB')
            noise_img = salt_and_pepper(img)
            noise_img.save(os.path.join(dest_dir, filename_without_extension + '_noise.jpg'))
            break
