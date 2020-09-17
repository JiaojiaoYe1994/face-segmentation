import cv2
from pylab import *
import torch
from torch import nn
from models import LinkNet34
import torchvision.transforms as transforms
from torch.autograd import Variable
import sys
from PIL import Image, ImageFilter

import cv2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = LinkNet34()
# model.load_state_dict(torch.load('linknet.pth'))
model.load_state_dict(torch.load('./model/linknet_0.pth', map_location=lambda storage, loc: storage))

model.eval()
model.to(device)

img_transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False)
])
t = transforms.Resize(size=(256,256))

def smooth_mask(mask):
    im = Image.fromarray(mask)
    im2 = im.filter(ImageFilter.MinFilter(3))
    im3 = im2.filter(ImageFilter.MaxFilter(5))
    return np.array(im3)

def plt_mask(path):
    _img = Image.open(path)
    a = img_transform(_img)
    a = a.unsqueeze(0)
    imgs = Variable(a.to(device))
    pred = model(imgs)
    img = np.array(_img)
    mask = pred > 0.5
    print(pred.max(), pred.min())
    mask = mask.squeeze()
    mask = mask.cpu().numpy()
    _img.save('000001.jpg')
    img=np.array(t(Image.fromarray(img)))
    img[mask==0]=0
    # plt.imshow(img)
    # plt.savefig('201752783_1_eval.jpg')
    Image.fromarray(np.uint8(img)).save('000001_eval.jpg')

# p = '/media/nasir/Drive1/datasets/test/0.jpeg'
p = '/data/jiaojiao/test/000001.jpg'
model.eval()
time_1 = time.time()

plt_mask(p)
time_2 = time.time()
print(f'time: {time_2 - time_1}')