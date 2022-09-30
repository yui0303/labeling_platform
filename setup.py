import os

dirs = ['prepare_data', 'log', 'txtArea']

for d in dirs:
    if not os.path.isdir(d):
        os.mkdir(d)

if not os.path.isdir('./prepare_data/with_annotation'):os.mkdir('./prepare_data/with_annotation')
if not os.path.isdir('./prepare_data/without_annotation'):os.mkdir('./prepare_data/without_annotation')
if not os.path.isdir('./prepare_data/with_annotation/train'):os.mkdir('./prepare_data/with_annotation/train')
if not os.path.isdir('./prepare_data/with_annotation/test'):os.mkdir('./prepare_data/with_annotation/test')
if not os.path.isdir('./prepare_data/without_annotation/train'):os.mkdir('./prepare_data/without_annotation/train')
if not os.path.isdir('./txtTemp'):os.mkdir('./txtTemp')