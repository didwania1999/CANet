# -*- coding: utf-8 -*-
"""DME with Val Set.ipynb


# Pre Processing

## Download DataSet
"""

# ========================== Create Google Drive Link ================================
IDRID_MESSIDOR = "https://drive.google.com/open?id=1UMHgqId4rO_N4E0fC4vKUk5wY8Ot6SB0"
IDRID = "https://drive.google.com/open?id=1Cz6wYV5xeWdn4oR6NWn7dhZZSGtq8TK2"
MESSIDOR = "https://drive.google.com/open?id=1CnICWI3XmcWgS9x8l36trTCxww9Nu9_s"
file_share_link = IDRID             # IDRID_MESSIDOR, IDRID, MESSIDOR
file_id = file_share_link[file_share_link.find("=") + 1:]
file_download_link = "https://docs.google.com/uc?export=download&confirm=Tqlj&id=" + file_id
file_name = "Dataset.zip"
print("Link Created !!")

# ==================== Download .RaR File =================================
# Dataset Size : 3.68G takes 1min 
! wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=$file_id' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$file_id" -O $file_name && rm -rf /tmp/cookies.txt

# ============= Extract .RaR and Delete extra folders ==============================
! pip install patool
import patoolib
import os 
patoolib.extract_archive("/content/Dataset.zip", outdir="/content/Dataset/")
print('Dataset Extracted')

! rm '/content/Dataset.zip'
! rm -r '/content/sample_data'
print('Folders Deleted')

# ============= Create Validation Dataset =====================
# Total Files 1613 takes 03:37min
! pip install split-folders
import splitfolders
splitfolders.ratio("/content/Dataset/content/Dataset/TrainSet/", output="/content/Dataset/content/Dataset/", seed=1337, ratio=(.8, .2), group_prefix=None)
print('Splited into TrainSet, ValSet & TestSet  !!')

! rm -r '/content/Dataset/content/Dataset/TrainSet'
! mv '/content/Dataset/content/Dataset/train' '/content/Dataset/content/Dataset/TrainSet'
! mv '/content/Dataset/content/Dataset/val' '/content/Dataset/content/Dataset/ValSet'
! mv '/content/Dataset/content/Dataset' '/content/Dataset/'
! rm -r '/content/Dataset/content/'
! mv '/content/Dataset/' '/content/dataset/'
! mv '/content/dataset/Dataset' '/content/'
! rm -r '/content/dataset/'
print('Done!!')

"""## GPU and CPU Enabling"""

# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 2.x
import tensorflow as tf
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

# Commented out IPython magic to ensure Python compatibility.
# Execution time for each cell
!pip install ipython-autotime
# %load_ext autotime

"""## Import Libraries & Hyper Parameters"""

# Commented out IPython magic to ensure Python compatibility.
# ============================== Import Libraries =========================
# %tensorflow_version 2.x
import tensorflow as tf
import os, os.path
import cv2
import h5py
import random
import keras
import json 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from keras import metrics
from keras.layers import Dense, Activation, Dropout
from keras.models import Model
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, VGG19,  Xception, ResNet50, ResNet152V2,InceptionResNetV2, InceptionV3, MobileNetV2, DenseNet201 
from google.colab.patches import cv2_imshow
from ipywidgets import IntProgress
from IPython.display import display
from tensorflow.keras import optimizers, metrics, losses
from numpy import std, mean, sqrt, max, min, exp

# =================  Hyper Parameters =================================
# train_path = '/content/Dataset/TrainSet'
# test_path = '/content/Dataset/TestSet'
# val_path = '/content/Dataset/ValSet'

train_path = '/content/Dataset/content/Dataset/train'
test_path = '/content/Dataset/content/Dataset/val'
Height =  4288
Width =  2848
Scale_Reduce = 9
Batch_Size = 32
NumberOfCategories = 3
MonitorValueName = 'precision'
Height = 224 #int(Height/Scale_Reduce)
Width = 224 #int(Width/Scale_Reduce)
Epoch = 500
Color_Scheme = cv2.COLOR_BGR2RGB
Channels = 3
#  VGG16, VGG19,  Xception, ResNet50, ResNet152V2, InceptionV3, InceptionResNetV2, MobileNetV2, DenseNet201,  EfficientNetB7 
ApplyModel = ResNet50
print("Height : "+str(Height))
print("Width : "+str(Width))

import glob 
path = train_path
class_dirs = glob.glob(path+"/*")
for class_name in class_dirs:
    no_of_images= len(glob.glob(class_name+"/**"))
    print("Class {} : {}".format(class_name, no_of_images))

"""## PreProcessing"""

from numpy import std, mean, log2, log
from math import cos, sin
from numpy import std, mean, sqrt, max, min, exp

def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def logTransform(img):
  img_mean=np.mean(np.mean(img))                           
  img_std= np.mean(np.std(img))          
  p=((img_mean/img_std)**1.15)                     
  ker1=np.array([[(1/np.log2(p**3)),(1/np.log2(p**2)),(1/np.log2(p**3))],[(1/np.log2(p**2)),(1/np.log2(p)),(1/np.log2(p**2))],[(1/np.log2(p**3)),(1/np.log2(p**2)),(1/np.log2(p**3))]], np.float32)
  ker2=np.array([[(1/np.log2(729)),(1/np.log2(81)),(1/np.log2(729))],[(1/np.log2(81)),(1/np.log2(9)),(1/np.log2(81))],[(1/np.log2(729)),(1/np.log2(81)),(1/np.log2(729))]], np.float32)
  ker3=np.array([[(1/np.log2(512)),(1/np.log2(64)),(1/np.log2(512))],[(1/np.log2(64)),(1/np.log2(8)),(1/np.log2(64))],[(1/np.log2(512)),(1/np.log2(64)),(1/np.log2(512))]], np.float32)
  img=cv2.filter2D(img,-1,ker2)
  return img

# Color Transormation
def ColorTransition(Simg, Timg):
    R, G, B=cv2.split(Simg)
    R = np.float64(R)
    G = np.float64(G)
    B = np.float64(B)
 
    R1, G1, B1=cv2.split(Timg)
    R1 = np.float64(R1)
    G1 = np.float64(G1)
    B1 = np.float64(B1)
 
    # conversion from RGB to lab color space -> Source Image
    L=0.3811*R+0.5783*G+0.0402*B;
    M=0.1967*R+0.7244*G+0.0782*B;
    S=0.0241*R+0.1288*G+0.8444*B;
    L = np.float64(L)
    M = np.float64(M)
    S = np.float64(S)
 
    # conversion from RGB to lab color space -> Target Image
    L1=0.3811*R1+0.5783*G1+0.0402*B1;
    M1=0.1967*R1+0.7244*G1+0.0782*B1;
    S1=0.0241*R1+0.1288*G1+0.8444*B1;
    L1 = np.float64(L1)
    M1 = np.float64(M1)
    S1 = np.float64(S1)
 
    I2 = cv2.merge((L,M,S))
    A2 = cv2.merge((L1,M1,S1))
    
    l=0.5774*L+0.5774*M+0.5774*S;
    a=0.4082*L+0.4082*M-0.8165*S;
    b=0.7071*L-0.7071*M;
    l = np.float64(l)
    a = np.float64(a)
    b = np.float64(b)
 
    l1=0.5774*L1+0.5774*M1+0.5774*S1;
    a1=0.4082*L1+0.4082*M1-0.8165*S1;
    b1=0.7071*L1-0.7071*M1;
    l1 = np.float64(l1)
    a1 = np.float64(a1)
    b1 = np.float64(b1)
 
    I3 = cv2.merge((l,a,b))
    A3 = cv2.merge((l1,a1,b1))
 
    std1=std(l1);
    std2=std(l);
 
    std3=std(a1);
    std4=std(a);
    
    std5=std(b1);
    std6=std(b);
 
    p=(sqrt(mean(l1))-(sqrt(mean(l))))/(sqrt(mean(l1))+(sqrt(mean(l))));   
    s=0
    if p>0:    
        s=0.9-(0.9 - 0.15)/(1+exp((p-0.45)/(0.05)));
    else:
        s=0.15;
 
    l2=mean(mean(l1))+(l-mean(mean(l)))*(1+s);
    a2=mean(mean(a1))+(a-mean(mean(a)));
    b2=mean(mean(b1))+(b-mean(mean(b)));
    l2 = np.float64(l2)
    a2 = np.float64(a2)
    b2 = np.float64(b2)
 
    l3=mean(mean(l1))+(l-mean(mean(l)))*(std1/std2);
    a3=mean(mean(a1))+(a-mean(mean(a)))*(std3/std4);
    b3=mean(mean(b1))+(b-mean(mean(b)))*(std5/std6);
    l3 = np.float64(l3)
    a3 = np.float64(a3)
    b3 = np.float64(b3)
 
    max_l=max(l2);
    min_l=min(l2);
    max_a=max(a2);
    min_a=min(a2);
    max_b=max(b2);
    min_b=min(b2);
  
    I4 = cv2.merge((l2,a2,b2))
 
    # Conversion from lab to RGB color space%
    L2=0.5774*l2+0.4082*a2+0.7071*b2;
    M2=0.5774*l2+0.4082*a2-0.7071*b2;
    S2=0.5774*l2-0.8169*a2;
    L2 = np.float64(L2)
    M2 = np.float64(M2)
    S2 = np.float64(S2)
 
    R2=4.4679*L2-3.5873*M2+0.1193*S2;
    G2=-1.2186*L2+2.3809*M2-0.1624*S2;
    B2=0.0497*L2-0.2439*M2+1.2045*S2;
 
    I5 = cv2.merge((R1, G2, B2))
    I6 = cv2.merge((l3,a3,b3))
 
    # Conversion from lab to RGB color space%
    L3=0.5774*l3+0.4082*a3+0.7071*b3;
    M3=0.5774*l3+0.4082*a3-0.7071*b3;
    S3=0.5774*l3-0.8169*a3;
    L3 = np.float64(L3)
    M3 = np.float64(M3)
    S3 = np.float64(S3)
 
    R3=4.4679*L3-3.5873*M3+0.1193*S3;
    G3=-1.2186*L3+2.3809*M3-0.1624*S3;
    B3=0.0497*L3-0.2439*M3+1.2045*S3;
    # R3 = np.float64(R3)
    # G3 = np.float64(G3)
    # B3 = np.float64(B3)
 
    I7 = cv2.merge((R3,G3,B3))
    I7 = normalize(I7)
    return I7

def rgb2lab(img):
    r, g, b = cv2.split(img)
    l = 0.3811*r + 0.5783*g + 0.0402*b;
    m = 0.1967*r + 0.7244*g + 0.0782*b;
    s = 0.0241*r + 0.1288*g + 0.8444*b;

    L = 0.5774*l + 0.5774*m + 0.5774*s;
    A = 0.4082*l + 0.4082*m - 0.8165*s;
    B = 0.7071*l - 0.7071*m;
    return L, A, B

def lab2rgb(l, a, b):
    L = 0.5774*l + 0.4082*a + 0.7071*b;
    M = 0.5774*l + 0.4082*a - 0.7071*b;
    S = 0.5774*l - 0.8169*a;

    R = 4.4679*L - 3.5873*M + 0.1193*S;
    G = -1.2186*L + 2.3809*M - 0.1624*S;
    B = 0.0497*L - 0.2439*M + 1.2045*S;
    return R, G, B

def sharpImage(img, x):
    kernel = np.array([[-x,-x,-x], [-x,9*x,-x], [-x,-x,-x]])
    img = cv2.filter2D(img, -1, kernel)
    return img

def edgeFilters(img):
    HEdges = np.array([[-1,0,1], [-2,0,2], [-1,0,1]])/4
    VEdges = np.array([[-1,-2,-1], [0,0,0], [1,2,1]])/4
    print(HEdges)
    print(VEdges)
    img = cv2.filter2D(img, -1, HEdges)
    img = cv2.filter2D(img, -1, VEdges)
    return img

def getfilter():
    p2 = np.array([15, 15]);
    alpha = 0.1;
    sum = 0;
    for theta in range(0, 120, 30):
        for sigma_x in np.arange(1., 3., 0.5):
            for sigma_y in np.arange(1, 3, 0.5):
                siz = (p2-1)/2
                a = cos(theta)**2/2/sigma_x**2+sin(theta)**2/2/sigma_y**2;
                b = -sin(2*theta)/4/sigma_x**2+sin(2*theta)/4/sigma_y**2;
                c = sin(theta)**2/2/sigma_x**2+cos(theta)**2/2/sigma_y**2;
                [x,y] = np.meshgrid(np.arange(-siz[1], siz[1], 1), np.arange(-siz[0], siz[0], 1));
                arg = (-(a*(x)**2+2*b*(x)*(y)+c*(y)**2));
                h=exp(arg);
                h=h*(1+log(sigma_x**alpha))*(1+log(sigma_y**alpha));
                lapl=((2*a*x+2*b*y)**2-2*a)+((2*c*y+2*b*x)**2-2*c);
                h1=h*lapl;
                h2=sum+h1;
    
    return h2

def preProcessingNew(Source):
    # Target = cv2.imread(train_path+'/0/IDRiD_061.jpg')
    Target = cv2.imread(test_path+'/0/20051020_44782_0100_PP.tif')
    Target = cv2.cvtColor(Target, cv2.COLOR_BGR2RGB)
    Target = cv2.resize(Target, (Height, Width))

    Sr, Sg, Sb = cv2.split(Source)
    Tr, Tg, Tb = cv2.split(Target)

    # conversion from RGB to lab color space    
    Sl, Sa, Sb = rgb2lab(Source)
    Tl, Ta, Tb = rgb2lab(Target)

    stdSl = std(Sl);
    stdTl = std(Tl);

    stdSa = std(Sa);
    stdTa = std(Ta);

    stdSb = std(Sb);
    stdTb = std(Tb);

    Fl = mean(mean(Tl))+(Sl-mean(mean(Sl)))*(stdTl/stdSl);
    Fa = mean(mean(Ta))+(Sa-mean(mean(Sa)))*(stdTa/stdSa);
    Fb = mean(mean(Tb))+(Sb-mean(mean(Sb)))*(stdTb/stdSb);

    Fr, Fg, Fb = lab2rgb(Fl, Fa, Fb);

    img = cv2.merge((Fr,Fg,Fb))
    img = normalize(img)
    partA = -10 * cv2.GaussianBlur(img, (21,21), 0)
    partB = 10*img + 0.5
    result = partA + partB
    result = normalize(result)

    Fr, Fg, Fb=cv2.split(result)

    x=5
    Mker=np.array([[1/(log2(x**3)), 1/(log2(x**2)), 1/(log2(x**3))],
         [1/(log2(x**2)), 1/(log2(x)), 1/(log2(x**2))],
         [1/(log2(x**3)), 1/(log2(x**2)), 1/(log2(x**3))]])

    
    # ker = getfilter()
    # Fg = cv2.filter2D(Fg, -1, Mker)
    # # Fg = cv2.filter2D(Fg, -1, ker)
    # Fg = abs(Fg)
    # Fg = normalize(sharpImage((1-Fg),1))
    # kernel = np.ones((3, 3), np.uint8) 
    # Fg = cv2.morphologyEx(Fg, cv2.MORPH_OPEN, kernel)
    return result

# Source = cv2.imread(test_path+'/2/IDRiD_035.jpg')

Source = cv2.imread(test_path+'/2/20051020_62385_0100_PP.tif')
Source = cv2.cvtColor(Source, cv2.COLOR_BGR2RGB)
Source = cv2.resize(Source, (Height, Width))

ProcessedImage = preProcessingNew(Source)
print(max(ProcessedImage), min(ProcessedImage)) 

# Displat Image
fig=plt.figure(dpi=150, facecolor='w', edgecolor='k')
plt.subplot(121)
plt.imshow(Source, cmap = 'gray')
plt.xticks([])
plt.yticks([])

plt.subplot(122)
plt.imshow(ProcessedImage, cmap = 'gray')
plt.xticks([])
plt.yticks([])

plt.show()

def clahe(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3, 3))
    cl1 = clahe.apply(img)
    return cl1

def Morphology(img, gray_img):
    kernel = np.ones((3, 3), np.uint8) 
    morph_img = cv2.morphologyEx(gray_img, cv2.MORPH_OPEN, kernel)
    result = cv2.bitwise_and(img, img, mask = morph_img)
    return result

def preProcessingwithClahe(img):
    # Target = cv2.imread(train_path+'/0/IDRiD_061.jpg')
    Target = cv2.imread(test_path+'/0/20051020_44782_0100_PP.tif')
    Target = cv2.cvtColor(Target, cv2.COLOR_BGR2RGB)
    Target = cv2.resize(Target, (Height, Width))
    # Color Normalization
    img = normalize(img)
    partA = -10 * cv2.GaussianBlur(img, (21,21), 0)
    partB = 10*img + 0.5
    result = partA + partB
    result = normalize(result)

    # Apply Clahe on each channel 
    r,g,b=cv2.split(result)
    gray_img = (((r+g+b)/3)*255).astype(np.uint8)
    green_img = clahe((g * 255).astype(np.uint8))
    red_img = clahe((r * 255).astype(np.uint8))
    blue_img = clahe((b * 255).astype(np.uint8))

    green_img = Morphology(green_img, gray_img)
    # red_img = Morphology(red_img, gray_img)
    # blue_img = Morphology(blue_img, gray_img)
    result1 = cv2.merge((red_img,green_img,blue_img))

    x=100
    Mker=np.array([[1/(log2(x**3)), 1/(log2(x**2)), 1/(log2(x**3))],
         [1/(log2(x**2)), 1/(log2(x)), 1/(log2(x**2))],
         [1/(log2(x**3)), 1/(log2(x**2)), 1/(log2(x**3))]])

    # green_img = cv2.filter2D(green_img, -1, Mker)
    green_img = sharpImage(green_img, 0.2)
    red_img = cv2.filter2D(red_img, -1, Mker)
    blue_img = cv2.filter2D(blue_img, -1, Mker)

    # Merge and Publish
    result2 = cv2.merge((blue_img,green_img,red_img))
    # result2 = 255 - result2
    return result2

Source = cv2.imread(test_path+'/2/20051020_62385_0100_PP.tif')
Source = cv2.cvtColor(Source, cv2.COLOR_BGR2RGB)
Source = cv2.resize(Source, (Height, Width))

# Apply Pre Processing with Clahe
result_img = preProcessingwithClahe(Source)

# Displat Image
fig=plt.figure(dpi=180, facecolor='w', edgecolor='k')
plt.subplot(121)
plt.imshow(Source, cmap = 'gray')
plt.xticks([])
plt.yticks([])

plt.subplot(122)
plt.imshow(result_img, cmap = 'gray')
plt.xticks([])
plt.yticks([])

plt.show()
fig.savefig("pre_process_image.png")

def preProcessing(img):
    # Target = cv2.imread(train_path+'/0/IDRiD_061.jpg')
    Target = cv2.imread(test_path+'/0/20051020_44782_0100_PP.tif')
    Target = cv2.cvtColor(Target, cv2.COLOR_BGR2RGB)
    Target = cv2.resize(Target, (Height, Width))
    img = ColorTransition(img, Target)
    img = normalize(img)
    partA = -4 * cv2.GaussianBlur(img, (25,25), 0)
    partB = 4*img + 0.5
    result = partA + partB
    result = normalize(result)
    return result

Source = cv2.imread(test_path+'/1/20051116_58835_0400_PP.tif')
Source = cv2.cvtColor(Source, cv2.COLOR_BGR2RGB) 
Source = cv2.resize(Source, (Height, Width))
print(np.max(Source), np.min(Source))
result_img = preProcessing(Source)

# Displat Image
fig=plt.figure(dpi=150, facecolor='w', edgecolor='k')
plt.subplot(121)
plt.imshow(Source, cmap = 'gray')
plt.xticks([])
plt.yticks([])

plt.subplot(122)
plt.imshow(result_img, cmap = 'gray')
plt.xticks([])
plt.yticks([])

plt.show()

from PIL import Image
from PIL import ImageOps
# ============================== Preprocessing ========================
def Preprocessing(img):
    Target = cv2.imread(train_path+'/0/IDRiD_061.jpg')
    Target = cv2.cvtColor(Target, cv2.COLOR_BGR2RGB)
    Target = cv2.resize(Target, (Height, Width))
    temp =  ColorTransition(img, Target) 
    r,g,b=cv2.split(temp)
    gray_img= (r+g+b)/3      
    green_img = 255-abs(g)
    mask = logTransform(green_img)

    # temp = cv2.bitwise_xor(img, img, mask = mask)
    return mask

# ================== Test ===========
Source = cv2.imread(test_path+'/2/IDRiD_035.jpg')
Source = cv2.cvtColor(Source, cv2.COLOR_BGR2RGB)
Source = cv2.resize(Source, (Height, Width))
fig=plt.figure(dpi=80, facecolor='w', edgecolor='k')
plt.subplot(121)
plt.imshow(Source)
img = preProcessing(Source)
plt.subplot(122)
plt.imshow(img, cmap = 'gray')
plt.show()
fig.savefig("pre_process_image.png")

"""## Load Function"""

def splitImage(img):
    Wmid = int(Width/2)
    Hmid = int(Height/2)
    img_part1 = img[0:Wmid,0:Hmid]
    img_part2 = img[0:Wmid,Hmid:]
    img_part3 = img[Wmid:,0:Hmid]
    img_part4 = img[Wmid:,Hmid:]
    return img_part1, img_part2, img_part3, img_part4

Source = cv2.imread(test_path+'/1/20051116_58835_0400_PP.tif')
Source = cv2.cvtColor(Source, cv2.COLOR_BGR2RGB) 
Source = cv2.resize(Source, (Height, Width))
result_img = splitImage(Source)

# Displat Image
fig=plt.figure(dpi=150, facecolor='w', edgecolor='k')
plt.subplot(221)
plt.imshow(result_img[0], cmap = 'gray')
plt.xticks([])
plt.yticks([])

plt.subplot(222)
plt.imshow(result_img[1], cmap = 'gray')
plt.xticks([])
plt.yticks([])

plt.subplot(223)
plt.imshow(result_img[2], cmap = 'gray')
plt.xticks([])
plt.yticks([])

plt.subplot(224)
plt.imshow(result_img[3], cmap = 'gray')
plt.xticks([])
plt.yticks([])

plt.show()

#Load the dataset
import pandas as pd 
def load_data(path, type='Train'):
    X_train = []
    Y_train = []
    fileData = []
    for sub in os.listdir(path):
        print('Progress for Class ',sub)
        temp = len(os.listdir(os.path.join(path, sub)))
        f = IntProgress(min=0, max=temp, step=1, description='Loading:', bar_style = 'info')
        display(f)
        for img in os.listdir(os.path.join(path, sub)):
            filepath = os.path.join(os.path.join(path, sub), img)
            temp = cv2.imread(filepath)
            temp = cv2.cvtColor(temp, Color_Scheme)
            temp = cv2.resize(temp, (Height, Width))
            temp = preProcessingwithClahe(temp)
            aug = [temp]
            if(type=='Train'):
                if int(sub) in [1, 2]:
                    aug = imageAugmentation(temp)
            
            for augimg in aug:
              X_train.append(augimg)
              category = to_categorical(int(sub), NumberOfCategories)
              fileData.append([img, int(sub), filepath])
              Y_train.append(category)
            
            f.value += 1
    fileData = pd.DataFrame(fileData, columns=['File Name', 'Category', 'File Size'])
    return np.asarray(X_train), np.asarray(Y_train), fileData

"""## Image Augmentation"""

# ================= Augmentation Methods =============================
    
def fill(img, h, w):
    img = cv2.resize(img, (h, w), cv2.INTER_CUBIC)
    return img
 
 
def horizontal_shift(img, ratio=0.0):
    if ratio > 1 or ratio < 0:
        print('Value should be less than 1 and greater than 0')
        return img
    ratio = random.uniform(-ratio, ratio)
    h, w = img.shape[:2]
    to_shift = w*ratio
    if ratio > 0:
        img = img[:, :int(w-to_shift), :]
    if ratio < 0:
        img = img[:, int(-1*to_shift):, :]
    img = fill(img, h, w)
    return img
 
def vertical_shift(img, ratio=0.0):
    if ratio > 1 or ratio < 0:
        print('Value should be less than 1 and greater than 0')
        return img
    ratio = random.uniform(-ratio, ratio)
    h, w = img.shape[:2]
    to_shift = h*ratio
    if ratio > 0:
        img = img[:int(h-to_shift), :, :]
    if ratio < 0:
        img = img[int(-1*to_shift):, :, :]
    img = fill(img, h, w)
    return img
 
def brightness(img, low, high):
    value = random.uniform(low, high)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype = np.float64)
    hsv[:,:,1] = hsv[:,:,1]*value
    hsv[:,:,1][hsv[:,:,1]>255]  = 255
    hsv[:,:,2] = hsv[:,:,2]*value 
    hsv[:,:,2][hsv[:,:,2]>255]  = 255
    hsv = np.array(hsv, dtype = np.uint8)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img
 
def zoom(img, value):
    if value > 1 or value < 0:
        print('Value for zoom should be less than 1 and greater than 0')
        return img
    value = random.uniform(value, 1)
    h, w = img.shape[:2]
    h_taken = int(value*h)
    w_taken = int(value*w)
    h_start = random.randint(0, h-h_taken)
    w_start = random.randint(0, w-w_taken)
    img = img[h_start:h_start+h_taken, w_start:w_start+w_taken, :]
    img = fill(img, h, w)
    return img
 
def channel_shift(img, value):
    value = int(random.uniform(-value, value))
    img = img + value
    img[:,:,:][img[:,:,:]>255]  = 255
    img[:,:,:][img[:,:,:]<0]  = 0
    img = img.astype(np.uint8)
    return img
 
def horizontal_flip(img, flag):
    if flag:
        return cv2.flip(img, 1)
    else:
        return img
 
def vertical_flip(img, flag):
    if flag:
        return cv2.flip(img, 0)
    else:
        return img
 
def both_flip(img, flag):
    if flag:
        return cv2.flip(cv2.flip(img, 0),1)
    else:
        return img
 
def rotation(img, angle):
    angle = int(random.uniform(-angle, angle))
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((int(w/2), int(h/2)), angle, 1)
    img = cv2.warpAffine(img, M, (w, h))
    return img

# ================== Augmentation Methods ====================
def imageAugmentation(img):
    result = []
    # result.append(img)

    # temp = horizontal_shift(img, 0.3)
    # result.append(temp)
 
    # temp = vertical_shift(img, 0.7)
    # result.append(temp)
 
    # temp = brightness(img, 0.5, 1.5)
    # result.append(temp)
 
    # temp = zoom(img, 0.5)
    # result.append(temp)
 
    # temp = channel_shift(img, 60)
    # result.append(temp)
 
    temp = horizontal_flip(img, True)
    result.append(temp)
 
    temp = vertical_flip(img, True)
    result.append(temp)
 
    temp = both_flip(img, True)
    result.append(temp)
 
    # temp = rotation(img, random.randint(1,360))
    # result.append(temp)

    # temp = rotation(img, random.randint(1,360))
    # result.append(temp)
 
    return result

#  ===================== Load Train, Test & Val Dataset =======================
print('Loading Training Data')
X_train, Y_train, Train_fileData = load_data(train_path, 'Train')
Train_fileData.to_csv('train.csv')

print('\nLoading Testing Data') 
X_test, Y_test, Test_fileData = load_data(test_path, 'Test')
Test_fileData.to_csv('test.csv')
 
# print('\nLoading Validation Data') 
# X_val, Y_val, Val_fileData = load_data(val_path, 'Train')
# Val_fileData.to_csv('val.csv')

print("Done Loading Data")
print("X_Train: {}\tY_Train: {}".format(len(X_train), len(Y_train)))
print("X_Test: {}\tY_Test: {}".format(len(X_test), len(Y_test)))
# print("X_Val: {}\tY_Val: {}\n".format(len(X_val), len(Y_val)))
 
print("Shape of Data")
print("X_Train: {}\tY_Train: {}".format(X_train.shape, Y_train.shape))
print("X_Test: {}\tY_Test: {}".format(X_test.shape, Y_test.shape))
# print("X_Val: {}\tY_Val: {}".format(X_val.shape, Y_val.shape))

def calculateImagesPerClass(Xclass):
    counts_per_class = [0] * NumberOfCategories
    for X in Xclass:
        temp = np.argmax(X)
        counts_per_class[temp]+=1
    
    for i in range(NumberOfCategories):
        print("In Class {} : {} images".format(i,counts_per_class[i]))
    print("\n")

print("Training Images Per Class")
calculateImagesPerClass(Y_train)

print("Testing Images Per Class")
calculateImagesPerClass(Y_test)

# print("Validation Images Per Class")
# calculateImagesPerClass(Y_val)

#================= Print Images ======================
import matplotlib.pyplot as plt
rows = 3
cols = 10
axes=[]
fig=plt.figure(figsize=(cols, rows), dpi=240, facecolor='w', edgecolor='k')
for a in range(rows*cols):
    axes.append(fig.add_subplot(rows, cols, a+1))
    plt.xticks([])
    plt.yticks([])   
    plt.imshow(X_train[180+a], cmap='gray')
fig.tight_layout()    
plt.show()
fig.savefig("Sample_Images.png")

"""## Spike Model"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Activation, Dense, Flatten, Dropout, Conv2D, MaxPooling2D, Input, Concatenate,Add, BatchNormalization
from tensorflow.keras.initializers import glorot_uniform
inputs = Input(shape=(Width, Height, Channels,))
X = Conv2D(filters=16,kernel_size=(3,3),padding="same", activation="relu")(inputs)
X = Conv2D(filters=32,kernel_size=(3,3),padding="same", activation="relu")(X)
X = MaxPooling2D(pool_size=(2,2),strides=(2,2))(X)
X = Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu")(X)
X = MaxPooling2D(pool_size=(2,2),strides=(2,2))(X)
X_shortcut = X

#Create Towers 
tower_1 = Conv2D(64, (1,1), padding='same', activation='relu')(X)
tower_1 = Conv2D(64, (3,3), padding='same', activation='relu')(tower_1)
tower_2 = Conv2D(64, (1,1), padding='same', activation='relu')(X)
tower_2 = Conv2D(64, (5,5), padding='same', activation='relu')(tower_2)
tower_3 = Conv2D(64, (1,1), padding='same', activation='relu')(X)
X = Concatenate(axis=3)([tower_1, tower_2, tower_3]) #Concatinate towers

# ResNet Skipconnections
X = Conv2D(filters = 64, kernel_size = (1, 1), strides = (1,1), padding = 'valid', kernel_initializer = glorot_uniform(seed=0))(X)
X = BatchNormalization(axis = 3)(X)
X = Activation('relu')(X)
X = Conv2D(filters = 64, kernel_size = (3, 3), strides = (1,1), padding = 'same', kernel_initializer = glorot_uniform(seed=0))(X)
X = BatchNormalization(axis = 3)(X)
X = Activation('relu')(X)
X = Conv2D(filters = 64, kernel_size = (1, 1), strides = (1,1), padding = 'valid', kernel_initializer = glorot_uniform(seed=0))(X)
X = BatchNormalization(axis = 3)(X)
X = Add()([X, X_shortcut ])
X = Activation('relu')(X)
# X = MaxPooling2D(pool_size=(2,2),strides=(2,2))(X)
X_shortcut2 = X

#Create Towers 
tower_1 = Conv2D(64, (1,1), padding='same', activation='relu')(X)
tower_1 = Conv2D(64, (3,3), padding='same', activation='relu')(tower_1)
tower_2 = Conv2D(64, (1,1), padding='same', activation='relu')(X)
tower_2 = Conv2D(64, (5,5), padding='same', activation='relu')(tower_2)
tower_3 = Conv2D(64, (1,1), padding='same', activation='relu')(X)
X = Concatenate(axis=3)([tower_1, tower_2, tower_3]) #Concatinate towers

# ResNet Skipconnections
X = Conv2D(filters = 64, kernel_size = (1, 1), strides = (1,1), padding = 'valid', kernel_initializer = glorot_uniform(seed=0))(X)
X = BatchNormalization(axis = 3)(X)
X = Activation('relu')(X)
X = Conv2D(filters = 64, kernel_size = (3, 3), strides = (1,1), padding = 'same', kernel_initializer = glorot_uniform(seed=0))(X)
X = BatchNormalization(axis = 3)(X)
X = Activation('relu')(X)
X = Conv2D(filters = 64, kernel_size = (1, 1), strides = (1,1), padding = 'valid', kernel_initializer = glorot_uniform(seed=0))(X)
X = BatchNormalization(axis = 3)(X)
X = Add()([X, X_shortcut2, X_shortcut ])
X = Activation('relu')(X)


# X = Flatten()(X)
X = Conv2D(filters=512,kernel_size=(3,3),padding="same", activation='relu')(X)
X = MaxPooling2D(pool_size=(3,3),strides=(2,2))(X)
X = Conv2D(filters=256,kernel_size=(3,3),padding="same", activation='relu')(X)
X = MaxPooling2D(pool_size=(3,3),strides=(2,2))(X)
X = Flatten()(X)
X = Dense(512)(X)
X = Dropout(0.576)(X)
X = Dense(256)(X)
X = Dropout(0.576)(X)

predictions = Dense(units=NumberOfCategories, activation="softmax")(X)
model = Model(inputs=inputs, outputs=predictions)
print(model.summary())
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

"""# Transfer Learning"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Activation, Dense, Flatten, Dropout, Conv2D, MaxPooling2D, Input, Concatenate,Add, BatchNormalization
from tensorflow.keras.initializers import glorot_uniform
from keras import  models
base_model = VGG16(weights='imagenet', include_top=False, pooling='avg')
# mark loaded layers as not trainable
for i, layer in enumerate(base_model.layers):
    if(i < 7):
        print(layer.name)
        layer.trainable = False
flat1 = Flatten()(base_model.layers[-1].output)
X = Dense(512)(flat1)
X = Dropout(0.576)(X)
X = Dense(256)(X)
X = Dropout(0.576)(X)
predictions = Dense(units=NumberOfCategories, activation="softmax")(X)
model = models.Model(inputs=base_model.input, outputs=predictions)

print(model.summary())
# plot_model(model_VGG16, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

"""# Define Models"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D
from tensorflow.keras.layers import Activation, Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, Dense, Flatten, UpSampling2D, Dropout, Conv2D, MaxPooling2D, Input, Concatenate,Add, BatchNormalization
from tensorflow.keras.initializers import glorot_uniform
from keras.layers import ReLU
from keras.layers import LeakyReLU
from tensorflow.keras import Input
from keras import backend as K 
from keras.layers import Layer 

class argMaxLayer(Layer): 
   def __init__(self, output_dim, **kwargs):
      self.output_dim = output_dim
      super(argMaxLayer, self).__init__(**kwargs) 
    
   def build(self, input_shape): 
    #   self.w = self.add_weight(name = 'kernel',
    #                   shape = (input_shape[-1],len(input_shape)),
    #                   initializer = 'normal',
    #                   trainable = True) 
    #   self.b = self.add_weight(name = 'kernel',
    #                   shape = (input_shape[-1],len(input_shape)),
    #                   initializer = 'normal',
    #                   trainable = True)
        # self.h = self.add_weight(name = 'kernel',
        #               shape = (1,len(input_shape)),
        #               initializer = 'normal',
        #               trainable = True) 
        super(argMaxLayer, self).build(input_shape) 

   def call(self, input_data):
        final_prediction = K.argmax(K.sum(input_data, axis=0))
        finalClass =  tf.one_hot( tf.cast(pred, tf.int32), self.output_dim, dtype=tf.float32)
        # hidden = self.h
        # score = tf.matmul(hidden, tf.cast(final_prediction, tf.float32))
        return finalClass

   def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
    
   def get_config(self):
        return {'output_dim': self.output_dim}


def upperLayer(inputs):
  X = Conv2D(filters=64,kernel_size=(5,5),padding="same", activation='relu')(inputs)
  X = Conv2D(filters=64,kernel_size=(3,3),padding="same", activation='relu')(X)
  X = MaxPooling2D(pool_size=(3,3),strides=(2,2))(X)
  X = Conv2D(filters=64,kernel_size=(3,3),padding="same", activation='relu')(X)
  X = MaxPooling2D(pool_size=(3,3),strides=(2,2))(X)
  X = Conv2D(filters=256,kernel_size=(3,3),padding="same", activation='relu')(X)
  X = MaxPooling2D(pool_size=(3,3),strides=(2,2))(X)
  return X

def model3():
    width = int(Width/2)
    height = int(Height/2)
    input1 =  keras.layers.Input(shape=(width, height, Channels,))
    input2 = keras.layers.Input(shape=(width, height, Channels,))
    input3 = keras.layers.Input(shape=(width, height, Channels,))
    input4 = keras.layers.Input(shape=(width, height, Channels,))
    X1 = upperLayer(input1)
    X2 = upperLayer(input2)
    X3 = upperLayer(input3)
    X4 = upperLayer(input4)
    X = Add()([X1, X2, X3, X4])
  
    X = Conv2D(filters=1024,kernel_size=(3,3),padding="same",kernel_initializer = glorot_uniform(seed=0),  activation='relu')(X)
    X = MaxPooling2D(pool_size=(3,3),strides=(2,2))(X)
    X = Conv2D(filters=512,kernel_size=(3,3),padding="same", kernel_initializer = glorot_uniform(seed=0), activation='relu')(X)
    X = MaxPooling2D(pool_size=(3,3),strides=(2,2))(X)
    X = Conv2D(filters=256,kernel_size=(3,3),padding="same", kernel_initializer = glorot_uniform(seed=0),  activation='relu')(X)
   
    X = Flatten()(X)
    X = Dense(1024)(X)
    X = Dropout(0.576)(X)
    X = Dense(512)(X)
    X = Dropout(0.576)(X)
    finalprediction = Dense(NumberOfCategories, activation="softmax")(X)
    
    model = Model(inputs=[input1, input2, input3, input4], outputs=finalprediction)
    return model

def model2():
    input = keras.layers.Input(shape=(Width, Height, Channels,))
    X = upperLayer(input)
    #Create Towers Line 1 
    L1_tower_1 = Conv2D(64, (1,1), padding='same', activation='relu')(X)
    L1_tower_1 = Conv2D(64, (3,3), padding='same', activation='relu')(L1_tower_1)
    L1_tower_2 = Conv2D(64, (1,1), padding='same', activation='relu')(X)
    L1_tower_2 = Conv2D(64, (5,5), padding='same', activation='relu')(L1_tower_2)
    L1_tower_3 = Conv2D(64, (1,1), padding='same', activation='relu')(X)
    Line1 = Concatenate(axis=3)([L1_tower_1, L1_tower_2, L1_tower_3])
    Line1 = MaxPooling2D(pool_size=(3,3),strides=(1,1))(Line1)
    Line1 = Conv2D(filters = 64, kernel_size = (1, 1), strides = (1,1), padding = 'valid', kernel_initializer = glorot_uniform(seed=0))(Line1)
    
    #Create Towers Line 2
    L2_tower_1 = Conv2D(128, (1,1), padding='same', activation='relu')(X)
    L2_tower_1 = Conv2D(128, (3,3), padding='same', activation='relu')(L2_tower_1)
    L2_tower_2 = Conv2D(128, (1,1), padding='same', activation='relu')(X)
    L2_tower_2 = Conv2D(128, (5,5), padding='same', activation='relu')(L2_tower_2)
    L2_tower_3 = Conv2D(128, (1,1), padding='same', activation='relu')(X)
    Line2 = Concatenate(axis=3)([L2_tower_1, L2_tower_2, L2_tower_3])
    Line2 = MaxPooling2D(pool_size=(3,3),strides=(1,1))(Line2)
    Line2 = Conv2D(filters = 64, kernel_size = (1, 1), strides = (1,1), padding = 'valid', kernel_initializer = glorot_uniform(seed=0))(Line2)

    #Create Towers Line 2
    L3_tower_1 = Conv2D(256, (1,1), padding='same', activation='relu')(X)
    L3_tower_1 = Conv2D(256, (3,3), padding='same', activation='relu')(L3_tower_1)
    L3_tower_2 = Conv2D(256, (1,1), padding='same', activation='relu')(X)
    L3_tower_2 = Conv2D(256, (5,5), padding='same', activation='relu')(L3_tower_2)
    L3_tower_3 = Conv2D(256, (1,1), padding='same', activation='relu')(X)
    Line3 = Concatenate(axis=3)([L3_tower_1, L3_tower_2, L3_tower_3])
    Line3 = MaxPooling2D(pool_size=(3,3),strides=(1,1))(Line3)
    Line3 = Conv2D(filters = 64, kernel_size = (1, 1), strides = (1,1), padding = 'valid', kernel_initializer = glorot_uniform(seed=0))(Line3)


    # X =  Concatenate(axis=3)([Line1, Line2, Line3])
    X = BatchNormalization(axis = 3)(Line2)
    X = Activation('relu')(X)
    X = Conv2D(filters = 64, kernel_size = (3, 3), padding = 'valid', kernel_initializer = glorot_uniform(seed=0))(X)
    X = MaxPooling2D(pool_size=(3,3),strides=(1,1))(X)
    X = Flatten()(X)
    X = Dense(1024, activation='relu')(X)
    X = Dropout(0.4)(X)
    X = Dense(512, activation='relu')(X)
    X = Dropout(0.4)(X)
    X = Dense(128, activation='relu')(X)
    X = Dropout(0.4)(X)
    finalprediction = Dense(NumberOfCategories, activation="softmax")(X)
    model = Model(inputs=input, outputs=finalprediction)
    return model

def defineHCNNModel():
    input = keras.layers.Input(shape=(Width, Height, Channels,))
    X1 = Conv2D(filters=32,kernel_size=(3,3),padding="same", kernel_initializer = glorot_uniform(seed=0), activation='relu')(input)
    X1 = Conv2D(filters=64,kernel_size=(3,3),padding="same", kernel_initializer = glorot_uniform(seed=0), activation='relu')(X1)
    X1 = MaxPooling2D(pool_size=(3,3),strides=(1,1))(X1)
    X1 = Conv2D(filters=128,kernel_size=(3,3),padding="same", kernel_initializer = glorot_uniform(seed=0), activation='relu')(X1)
    X1 = MaxPooling2D(pool_size=(3,3),strides=(1,1))(X1)
    X1 = Conv2D(filters=512,kernel_size=(3,3),padding="same",kernel_initializer = glorot_uniform(seed=0), activation='relu')(X1)
    X1 = MaxPooling2D(pool_size=(3,3),strides=(1,1))(X1)
    X1 = Conv2D(filters=64,kernel_size=(3,3),padding="same", kernel_initializer = glorot_uniform(seed=0), activation='relu')(X1)
    X1 = MaxPooling2D(pool_size=(3,3),strides=(1,1))(X1)
    X1 = Flatten()(X1)
    X1 = Dense(1024)(X1)
    X1 = Dropout(0.4)(X1)
    X1 = Dense(512)(X1)
    X1 = Dropout(0.4)(X1)
    predictions1 = Dense(256)(X1)

    X2 = Conv2D(filters=32,kernel_size=(5,5),padding="same",kernel_initializer = glorot_uniform(seed=0), activation='relu')(input)
    X2 = Conv2D(filters=64,kernel_size=(5,5),padding="same",kernel_initializer = glorot_uniform(seed=0), activation='relu')(X2)
    X2 = MaxPooling2D(pool_size=(5,5),strides=(1,1))(X2)
    X2 = Conv2D(filters=128,kernel_size=(5,5),padding="same",kernel_initializer = glorot_uniform(seed=0), activation='relu')(X2)
    X2 = MaxPooling2D(pool_size=(5,5),strides=(1,1))(X2)
    X2 = Conv2D(filters=64,kernel_size=(3,3),padding="same",kernel_initializer = glorot_uniform(seed=0), activation='relu')(X2)
    X2 = MaxPooling2D(pool_size=(5,5),strides=(1,1))(X2)
    X2 = Conv2D(filters=64,kernel_size=(3,3),padding="same",kernel_initializer = glorot_uniform(seed=0), activation='relu')(X2)
    X2 = MaxPooling2D(pool_size=(5,5),strides=(1,1))(X2)
    X2 = Flatten()(X2)
    X2 = Dense(1024)(X2)
    X2 = Dropout(0.4)(X2)
    X2 = Dense(512)(X2)
    X2 = Dropout(0.4)(X2)
    predictions2 = Dense(256)(X2)

    X3 = Conv2D(filters=32,kernel_size=(7,7),padding="same",kernel_initializer = glorot_uniform(seed=0), activation='relu')(input)
    X3 = Conv2D(filters=64,kernel_size=(7,7),padding="same",kernel_initializer = glorot_uniform(seed=0), activation='relu')(X3)
    X3 = MaxPooling2D(pool_size=(7,7),strides=(1,1))(X3)
    X3 = Conv2D(filters=128,kernel_size=(7,7),padding="same",kernel_initializer = glorot_uniform(seed=0), activation='relu')(X3)
    X3 = MaxPooling2D(pool_size=(7,7),strides=(1,1))(X3)
    X3 = Conv2D(filters=512,kernel_size=(3,3),padding="same",kernel_initializer = glorot_uniform(seed=0), activation='relu')(X3)
    X3 = MaxPooling2D(pool_size=(7,7),strides=(1,1))(X3)
    X3 = Conv2D(filters=64,kernel_size=(7,7),padding="same",kernel_initializer = glorot_uniform(seed=0), activation='relu')(X3)
    X3 = MaxPooling2D(pool_size=(7,7),strides=(1,1))(X3)
    X3 = Flatten()(X3)
    X3 = Dense(1024, activation='relu')(X3)
    X3 = Dropout(0.4)(X3)
    X3 = Dense(512)(X3)
    X3 = Dropout(0.4)(X3)
    predictions3 = Dense(256)(X3)

    X = Concatenate(axis=1)([predictions1,predictions2,predictions3])
    X = Dense(128)(X)
    X = Dropout(0.4)(X)
    finalprediction = Dense(NumberOfCategories, activation="softmax")(X)

    model = Model(inputs=input, outputs=finalprediction)
    return model

def defineModel():
  input = keras.layers.Input(shape=(Width, Height, Channels,))
  upperlayer = upperLayer(input)
  X = BatchNormalization(axis = 3)(upperlayer)
  X = Conv2D(filters=1024,kernel_size=(3,3),padding="same",kernel_initializer = glorot_uniform(seed=0),  activation='relu')(X)
  X = MaxPooling2D(pool_size=(3,3),strides=(2,2))(X)
  X = Conv2D(filters=512,kernel_size=(3,3),padding="same", kernel_initializer = glorot_uniform(seed=0), activation='relu')(X)
  X = MaxPooling2D(pool_size=(3,3),strides=(2,2))(X)
  X = Conv2D(filters=256,kernel_size=(3,3),padding="same", kernel_initializer = glorot_uniform(seed=0),  activation='relu')(X)
  X = Flatten()(X)
  X = Dense(1024)(X)
  X = Dropout(0.576)(X)
  X = Dense(512)(X)
  X = Dropout(0.576)(X)
  predictions = Dense(units=NumberOfCategories, activation="softmax")(X)
  model = Model(inputs=input, outputs=predictions)
  return model

def HybridModel():
    base_model = VGG19(weights='imagenet')
    # base_model.trainable = False
    X = base_model.layers[5].output
    X = Conv2D(filters=512,kernel_size=(3,3),padding="same", activation='relu')(X)
    X = MaxPooling2D(pool_size=(3,3),strides=(2,2))(X)
    X = Conv2D(filters=256,kernel_size=(3,3),padding="same", activation='relu')(X)
    X = MaxPooling2D(pool_size=(3,3),strides=(2,2))(X)
    X = Flatten()(X)
    X = Dense(512)(X)
    X = Dropout(0.576)(X)
    X = Dense(256)(X)
    X = Dropout(0.576)(X)
    predictions = Dense(units=NumberOfCategories, activation="softmax")(X)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

#======================= Pre-trained Model ==================================== 
# input1 = keras.layers.Input(shape=(Width, Height, Channels, ))
from keras.utils.vis_utils import plot_model
model = defineModel()
print(model.summary())
# plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

"""# Loss Functions and Model Def"""

def specificity(y_pred, y_true):
    neg_y_true = 1 - y_true
    neg_y_pred = 1 - y_pred
    fp = K.sum(neg_y_true * y_pred)
    tn = K.sum(neg_y_true * neg_y_pred)
    specificity = tn / (tn + fp + K.epsilon())
    return specificity

def sensitivity(y_pred, y_true):
    neg_y_true = 1 - y_true
    neg_y_pred = 1 - y_pred
    tp = K.sum(y_true * y_pred)
    fn = K.sum(neg_y_pred * y_true)
    sensitivity = tp / (tp + fn + K.epsilon())
    return sensitivity

def f1_score(y_true, y_pred):
    precision = specificity(y_true, y_pred)
    recall = sensitivity(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

!pip install tensorflow-addons

# ============================= Create Compiler ============================= 
from tensorflow.keras import optimizers, metrics, losses
import tensorflow_addons as tfa
adam = optimizers.Adam(learning_rate=0.001)
adagrad = optimizers.Adagrad(learning_rate=0.001)
opt = tfa.optimizers.RectifiedAdam(lr=1e-3)
# losses.CategoricalCrossentropy()
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# model.compile(optimizer=adagrad,
            #   loss= ['mse'],
            #   metrics=[ 'accuracy', keras.metrics.categorical_accuracy,tf.keras.metrics.TopKCategoricalAccuracy(k=5), specificity, sensitivity, f1_score ])

print('Compiled!')

# =============================== Calculate Steps per Epoch ===========================
def cal_step(val, batch_size):
  batch = int(len(val)/batch_size) 
  return batch

print("Batch Size: "+ str(Batch_Size))
Train_Batches = cal_step(X_train,Batch_Size)
Test_Batches = cal_step(X_test,Batch_Size) 
# Val_Batches = cal_step(X_val,Batch_Size) 
print("Train Batches: {}".format(Train_Batches))
print("Test Batches: {}".format(Test_Batches))
# print("Val Batches: {}".format(Val_Batches))

"""## Optimize Hyperparameters"""

# ================= Install Optuna =====================
! pip install optuna
import optuna
print('Optuna Version: '+str(optuna.__version__))

def create_model(trial):
    dropout_Value = trial.suggest_float("weight_decay", 0.1, 0.6, log=True)
    input = keras.layers.Input(shape=(Width, Height, Channels,))
    X = upperLayer(input)
    X = Conv2D(filters=1024,kernel_size=(3,3),padding="same", activation='relu')(X)
    X = Conv2D(filters=512,kernel_size=(3,3),padding="same", activation='relu')(X)
    X = Conv2D(filters=256,kernel_size=(3,3),padding="same", activation='relu')(X)
    X = Flatten()(X)
    X = Dense(1024)(X)
    X = Dropout(dropout_Value)(X)
    X = Dense(512)(X)
    X = Dropout(dropout_Value)(X)
    X = Dense(256)(X)
    X = Dropout(dropout_Value)(X)
    predictions = Dense(units=NumberOfCategories, activation="softmax")(X)
    model = Model(inputs=input, outputs=predictions)
    return model

def create_optimizer(trial):
    kwargs = {}
    optimizer_options = ["RMSprop", "Adam", "SGD"]
    optimizer_selected = trial.suggest_categorical("optimizer", optimizer_options)
    if optimizer_selected == "RMSprop":
        kwargs["learning_rate"] = trial.suggest_float(
            "rmsprop_learning_rate", 1e-5, 1e-1, log=True
        )
        kwargs["decay"] = trial.suggest_float("rmsprop_decay", 0.85, 0.99)
        kwargs["momentum"] = trial.suggest_float("rmsprop_momentum", 1e-5, 1e-1, log=True)
    elif optimizer_selected == "Adam":
        kwargs["learning_rate"] = trial.suggest_float("adam_learning_rate", 1e-5, 1e-1, log=True)
    elif optimizer_selected == "SGD":
        kwargs["learning_rate"] = trial.suggest_float(
            "sgd_opt_learning_rate", 1e-5, 1e-1, log=True
        )
        kwargs["momentum"] = trial.suggest_float("sgd_opt_momentum", 1e-5, 1e-1, log=True)

    optimizer = getattr(tf.optimizers, optimizer_selected)(**kwargs)
    return optimizer

# ================== Objective Function =======================
from tensorflow.keras import optimizers, metrics, losses
def objective(trial):
    model = create_model(trial)
    optimizer = create_optimizer(trial)
    model.compile(optimizer, loss=[losses.CategoricalCrossentropy()], metrics=[metrics.categorical_accuracy])
    with tf.device("/device:GPU:0"):
        my_callbacks = [
            keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', patience=5),
        ]

        history = model.fit(
                            X_train, Y_train, 
                            validation_data = (X_val, Y_val),
                            batch_size=Batch_Size, 
                            epochs=Epoch,
                            shuffle = True, 
                            verbose = 1,
                            validation_steps = cal_step(X_val,Batch_Size),
                            steps_per_epoch = cal_step(X_train,Batch_Size),
                            class_weight = class_weights, 
                            callbacks = my_callbacks
                        )
        y_pred = model.predict(X_test, batch_size=Batch_Size, verbose=1)
        true = 0
        for i in range(len(y_pred)):
            if np.argmax(y_pred[i]) == np.argmax(Y_test[i]):
                true+=1
        accuracy = true/len(y_pred)

    return accuracy

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=25)
print("Number of finished trials: ", len(study.trials))

print("Best trial:")
trial = study.best_trial
print("  Value: ", trial.value)
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

print("Best trial:")
trial = study.best_trial
print("  Value: ", trial.value)
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

# =============== All Trails ===================
import pandas as pd
print('No. of Trails: '+str(len(study.trials)))
df = study.trials_dataframe()
print(df)

# Save Dataframe
df.to_csv('Otimization.csv')

"""## Run Model"""

#======================= Calculate Weight of Each Class ===========================
from sklearn.utils import class_weight
import numpy as np
class_weights = class_weight.compute_class_weight('balanced', np.unique(np.argmax(Y_train, axis=1)), np.argmax(Y_train, axis=1))
temp = {}

for key,value in enumerate(class_weights):
    temp[key]= value

class_weights = temp
print(class_weights)

from keras import backend as K
import keras 
session = keras.backend.get_session()
init = tf.global_variables_initializer()
session.run(init)

class StepDecay(LearningRateDecay):
	def __init__(self, initAlpha=0.01, factor=0.25, dropEvery=10):
		# store the base initial learning rate, drop factor, and
		# epochs to drop every
		self.initAlpha = initAlpha
		self.factor = factor
		self.dropEvery = dropEvery
	def __call__(self, epoch):
		# compute the learning rate for the current epoch
		exp = np.floor((1 + epoch) / self.dropEvery)
		alpha = self.initAlpha * (self.factor ** exp)
		# return the learning rate
		return float(alpha)
  
class PolynomialDecay(LearningRateDecay):
	def __init__(self, maxEpochs=100, initAlpha=0.01, power=1.0):
		# store the maximum number of epochs, base learning rate,
		# and power of the polynomial
		self.maxEpochs = maxEpochs
		self.initAlpha = initAlpha
		self.power = power
	def __call__(self, epoch):
		# compute the new learning rate based on polynomial decay
		decay = (1 - (epoch / float(self.maxEpochs))) ** self.power
		alpha = self.initAlpha * decay
		# return the new learning rate
		return float(alpha)

from keras.callbacks import LearningRateScheduler
import math

def step_decay(epoch):
   initial_lrate = 0.001
   drop = 0.5
   epochs_drop = 10.0
   lrate = initial_lrate * math.pow(drop,  math.floor((1+epoch)/epochs_drop))
   print("Learning Rate: "+str(lrate))
   return lrate

lrate = LearningRateScheduler(step_decay)

def toTupple(X):
    return [X[:,0], X[:,1], X[:,2], X[:,3]]

pip install scikit-plot

import os

from keras.callbacks import Callback
import matplotlib.pyplot as plt
import numpy as np
from scikitplot.metrics import plot_confusion_matrix, plot_roc
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report


class PerformanceVisualizationCallback(Callback):
    def __init__(self, model, validation_data, image_dir):
        super().__init__()
        self.model = model
        self.validation_data = validation_data
        os.makedirs(image_dir, exist_ok=True)
        self.image_dir = image_dir

    def on_epoch_end(self, epoch, logs={}):
        y_pred = np.asarray(self.model.predict(self.validation_data[0]))
        y_true = self.validation_data[1]             
        y_pred_class = np.argmax(y_pred, axis=1)

        # plot and save confusion matrix
        fig, ax = plt.subplots(figsize = (NumberOfCategories, NumberOfCategories))
        # plot_confusion_matrix(y_true, y_pred_class, ax=ax)
        results = confusion_matrix(y_true, y_pred_class)
        sn.set(font_scale=1)
        map = sn.heatmap(results, annot=True,annot_kws={"size": 18}, center=0,cmap="YlGnBu", fmt='.1f',lw=0.5, cbar=True, linewidths=2, linecolor='black', cbar_kws={'label': '# Images', 'orientation': 'horizontal'})
        map.set_title('Confusion matrix of {}'.format(epoch))
        map.set_xticklabels(class_labels, fontsize = 10)
        map.set_yticklabels(class_labels, fontsize = 10)
        fig = map.get_figure() 
        fig.savefig(os.path.join(self.image_dir, f'confusion_matrix_epoch_{epoch}'))

       # plot and save roc curve
        fig, ax = plt.subplots(figsize=(3,3))
        plot_roc(y_true, y_pred, ax=ax)
        fig.savefig(os.path.join(self.image_dir, f'roc_curve_epoch_{epoch}'))

performance_cbk = PerformanceVisualizationCallback(
                      model=model,
                      validation_data=(X_test, Y_test),
                      image_dir='performance_vizualizations')
#  =================================== Run the Model ====================================
import datetime
logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
my_callbacks = [
    # lrate,  
    # performance_cbk,
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=100),
    # keras.callbacks.ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}-{val_accuracy:.3f}.h5'),
    tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1),
    keras.callbacks.ModelCheckpoint(filepath='modelA.h5', mode='max', monitor='val_categorical_accuracy', verbose=2, save_best_only=True),
    keras.callbacks.ModelCheckpoint(filepath='modelS.h5', mode='max', monitor='val_sensitivity', verbose=2, save_best_only=True),
    # keras.callbacks.ModelCheckpoint(filepath='modelS.h5', mode='max', monitor='val_sensitivity', verbose=2, save_best_only=True),
]
 
history = []

with tf.device('/device:GPU:0'):
    history = model.fit(
                        X_train, Y_train, 
                        validation_data = (X_test, Y_test),
                        # batch_size=Batch_Size, 
                        epochs= Epoch,
                        shuffle = True, 
                        verbose = 1,
                        validation_steps = cal_step(X_test,Batch_Size),
                        steps_per_epoch = cal_step(X_train,Batch_Size),
                        class_weight = class_weights, 
                        callbacks = my_callbacks
                    )

# Load Best Model
model.load_weights("modelS.h5")
print("Model Loaded!!")

# acc = history.history['val_categorical_accuracy']
# loss = history.history['val_loss']
# print('/nfinal precision',acc)
# print('final loss/n',loss)
 
model.load_weights("modelS.h5")
# compute the Test Accuracy
y_pred = model.predict(X_test)
true = 0
print("Total Predicted Images {}".format(len(y_pred)))
for i in range(len(y_pred)):
    if np.argmax(y_pred[i]) == np.argmax(Y_test[i]):
        true+=1
print("Test Accuracy:", true/len(y_pred))

with open('history.npy', 'wb') as f:
    np.save(f, np.array(acc))
    np.save(f, np.array(loss))
    np.save(f,np.array(true/len(y_pred)))

np.argmax(y_pred[1])

import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='training data')
plt.plot(history.history['val_loss'], label='testing data')
plt.ylabel('Loss value')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.savefig("plot_Loss.png")
plt.show()

plt.plot(history.history['categorical_accuracy'], label='training data')
plt.plot(history.history['val_categorical_accuracy'], label='testing data')
plt.ylabel('val_categorical_accuracy')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.savefig("plot_Acc.png")
plt.show()

from numpy import argmax
import pandas as pd
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report
import seaborn as sn
import numpy as np

# Load Best Model
model.load_weights("modelS.h5")
print("Model Loaded!!")

class_labels = [ i for i in range(NumberOfCategories)]
y_prob = model.predict(X_test, verbose=1, steps=cal_step(X_test,Batch_Size)) 
predicted = np.argmax(y_prob,axis=1)
actual = np.argmax(Y_test,axis=1)
# actual = Y_test

results = confusion_matrix(actual, predicted)
acc = accuracy_score(actual, predicted) 
report = classification_report(actual, predicted)
print(results) 
print ('Accuracy Score :',acc) 
print ('Report : ', report)

sn.set(font_scale=1)
plt.figure(figsize = (NumberOfCategories, NumberOfCategories))
map = sn.heatmap(results, annot=True,annot_kws={"size": 18}, center=0,cmap="YlGnBu", fmt='.1f',lw=0.5, cbar=True, linewidths=2, linecolor='black', cbar_kws={'label': '# Images', 'orientation': 'horizontal'})
map.set_title('Confusion matrix')
map.set_xticklabels(class_labels, fontsize = 10)
map.set_yticklabels(class_labels, fontsize = 10)
figure = map.get_figure() 

figure.savefig("plot_Conf.png")
with open('report.npy', 'wb') as f:
    np.save(f, np.array(results))
    np.save(f, np.array(acc))
    np.save(f, np.array(report))



# List of Wrong Prediction
def getWrongImages(y_pred, y_true, filedata):
    temp = y_pred != y_true
    print("Wrong Predicted : " + str(np.sum(temp)))
    return str(np.sum(temp)), X_test[temp==True], y_true[temp==True], y_pred[temp==True], filedata[temp==True]['File Name'].values
result = getWrongImages(predicted, actual, Test_fileData)
print(result[3])

# List of Wrong Prediction
def getWrongImages(y_pred, y_true, filedata):
    temp = y_pred != y_true
    print("Wrong Predicted : " + str(np.sum(temp)))
    return str(np.sum(temp)), X_test[temp==True], y_true[temp==True], y_pred[temp==True], filedata[temp==True]['File Name'].values
    
def fun(result_tupple):
    #================= Print Images ======================
    rows = int(result_tupple[0])
    cols = 4
    axes=[]
    X_train = result_tupple[1]
    acc = result_tupple[2]
    pre = result_tupple[3]
    filedata = result_tupple[4]

    fig=plt.figure(figsize=(5,rows), dpi=120, facecolor='w', edgecolor='k')
    for a in range(rows):
        axes.append(fig.add_subplot(rows, cols, a*cols+1))
        plt.xticks([])
        plt.yticks([])
        plt.text(0.5, 0.5, str(filedata[a]), horizontalalignment='center', verticalalignment='center', fontsize=7)

        axes.append(fig.add_subplot(rows, cols, a*cols+2))
        plt.xticks([])
        plt.yticks([])   
        plt.imshow(X_train[a], cmap='gray')

        axes.append(fig.add_subplot(rows, cols, a*cols+3))
        plt.xticks([])
        plt.yticks([])
        plt.text(0.5, 0.5, "Actual: "+str(acc[a]),horizontalalignment='center', verticalalignment='center', fontsize=12)

        axes.append(fig.add_subplot(rows, cols, a*cols+4))
        plt.xticks([])
        plt.yticks([])
        plt.text(0.5, 0.5, "Pred.: "+str(pre[a]), horizontalalignment='center', verticalalignment='center', fontsize=12)

    fig.tight_layout()    
    plt.show()
    fig.savefig("falsePredict.png")

result = getWrongImages(predicted, actual, Test_fileData)
fun(result)

#================= Print Images ======================
import matplotlib.pyplot as plt
rows = 4
cols = 4
axes=[]
fig=plt.figure(figsize=(cols, rows), dpi=140, facecolor='w', edgecolor='k')
for a in range(rows*cols):
    axes.append(fig.add_subplot(rows, cols, a+1))
    plt.xticks([])
    plt.yticks([])   
    plt.imshow(result[1][a], cmap='gray')
    # plt.text(0, 0, str(result[1][a]), horizontalalignment='right', verticalalignment='bottom', fontsize=8)
    # plt.text(10, 0, str(result[2][a]), horizontalalignment='left', verticalalignment='bottom', fontsize=8)
fig.tight_layout()    
plt.show()
fig.savefig("falseSampleImages.png")

from sklearn.metrics import roc_curve, auc
y_pred = model.predict(X_test).ravel()
fpr, tpr, thresholds = roc_curve(Y_test.ravel(), y_pred)
auc = auc(fpr, tpr)

fig = plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='(area = {:.3f})'.format(auc))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()
fig.savefig('roc_curve')

import zipfile

file_name = "Experiment_MESSIDOR_IA_3.zip"

with zipfile.ZipFile(file_name, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    zf.write("history.npy")
    # zf.write("model.h5")
    zf.write("model_plot.png")
    zf.write("plot_Acc.png")
    zf.write("plot_Conf.png")
    zf.write("plot_Loss.png")
    zf.write("report.npy")
    # zf.write("pre_process_image.png")
    zf.write("falseSampleImages.png")
    zf.write("falsePredict.png")
    zf.write("roc_curve.png")
    zf.write("train.csv")
    zf.write("test.csv")
    # zf.write("val.csv")

print("Zip Done -> Experiment Saved!!!")

from keras import backend as K
import numpy as np

layer_of_interest=0
intermediate_tensor_function = K.function([model.layers[0].input],[model.layers[layer_of_interest].output])
intermediate_tensor = intermediate_tensor_function([X.iloc[0,:].values.reshape(1,-1)])[0]

