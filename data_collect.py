from PIL import Image
import glob
import os
import numpy as np
from tqdm import tqdm

root = '/data/jiaojiao/CelebAMask-HQ/'

save_dir = root+'masks/'

if not os.path.exists(save_dir):
    os.mkdir(root+'masks')

mask_files = glob.glob(root+ 'masks_png/*')

# rename *_skin.png to *.bmp
# for f_p in mask_files:
#     name = f_p.split('/')[-1]
#     name1 = name.replace('_skin','')
#     name_ = name1.replace('png', 'bmp')
#     save_path = save_dir + name_
#     # print(save_path,name_)
#     img = Image.open(f_p)
#     img_ = Image.fromarray(np.array(img)[...,0])
#     img_.save(save_path)

# img_dir = root + 'images'
# img_files = glob.glob(root+ 'images/*')
# save_dir = root + 'images_pend/'
# if not os.path.exists(save_dir):
#     os.mkdir(save_dir)
#
# for f_p in img_files:
#     name = f_p.split('/')[-1]
#     name1 =name.replace('.jpg', '')
#     name_ = '%05d' % (int(name1))
#     save_path = save_dir + name_ +'.jpg'
#     # print(save_path,name_)
#     img = Image.open(f_p)
#     img.save(save_path)

def dataset_split(data_dir):
    save_dir = root + 'masks/'

    files_path = glob.glob(root + 'masks/*bmp')
    files_names = [x.replace(root + 'masks/', '') for x in files_path]

    # split data, 90% train, 10% validation
    train_limit = (9 * len(files_names)) // 10

    train_files_names = files_names[0:train_limit]
    val_files_names = files_names[train_limit:]

    print('Train dataset size: ', len(train_files_names), ' | Validation dataset size:', len(val_files_names))

    if not os.path.exists(root+ 'train/images'):
        os.mkdir(root+'train/')
        os.mkdir(root+'train/images/')
        os.mkdir(root+'train/masks/')
    if not os.path.exists(root+ 'val/images'):
        os.mkdir(root+'val/')
        os.mkdir(root+'val/images/')
        os.mkdir(root+'val/masks/')

    for file_name in tqdm(train_files_names):
        mask_path = root + 'masks/' + file_name
        img_path = root + 'images/' + file_name.replace('bmp', 'jpg')
        img = Image.open(img_path)
        mask = Image.open(mask_path)
        img.save(root+'train/images/'+file_name.replace('bmp', 'png'))
        mask.save(root+'train/masks/'+file_name.replace('bmp', 'png'))

    for file_name in tqdm(val_files_names):
        mask_path = root + 'masks/' + file_name
        img_path = root + 'images/' + file_name.replace('bmp', 'jpg')
        img = Image.open(img_path)
        mask = Image.open(mask_path)
        img.save(root+'val/images/'+file_name.replace('bmp', 'png'))
        mask.save(root+'val/masks/'+file_name.replace('bmp', 'png'))

    print('Dataset split done')

dataset_split(root)