'''
This script works for dataset preparation. The dataset used in this project comes from multiple source
and each dataset has different format. We process each dataset accordingly and finally combine them to the final
dataset.

Dataset Source:
1. CelebA: https://www.kaggle.com/jessicali9530/celeba-dataset
2. Exemplar-Based Face Parsing: http://pages.cs.wisc.edu/~lizhang/projects/face-parsing/
3. Labeled Faces in the Wild: http://vis-www.cs.umass.edu/lfw/

The final train dataset consists of the following folder:
/images
/masks

'''


import os
import glob
import sys
import cv2
from PIL import Image
import numpy as np


root = '~/SmithCVPR2013_dataset_resized'

# img_pth = os.path.join(root, 'images')
# label_dir = os.path.join(root, 'labels')

# mask_dir = '/home/jiaojiao/jiaojiao/project/dubhe/face-segmentation/datasets/SmithCVPR2013_dataset_resized/masks'

label_pth = glob.glob(label_pth+'/*')

print('Dataset size: ', len(labels_files))
label = glob.glob(labels_files[0]+'/*[1-9].png')


def helen_combine_seg(path):
    label = glob.glob(path + '/*[1-9].png')
    name = label[0].split('/')[-2]

    mask = np.zeros(np.array(Image.open(label[0])).shape)
    for f_p in label:
        bmp = Image.open(f_p)
        #     bmp.show()
        mask += np.array(bmp)

    mask = Image.fromarray(np.uint8(mask))

    save_path = os.path.join(mask_dir, name + '.bmp')

    mask.save(save_path)

def helen_process(root):
    label_dir = os.path.join(root, 'labels')

    mask_dir = os.path.join(root, 'masks')

    label_pth = glob.glob(label_dir + '/*')

    for p in labels_pth:
        helen_combine_seg(p)


def main():
    helen_process(root)

if __name__ == "__main__":
    main()