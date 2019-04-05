import os
import cv2
import numpy as np 
from preprocessing import parse_annotation
from frontend import *

images_folder="/home/yolo2/RBC_datasets/JPEGImages/"
annotations_folder="/home/yolo2/RBC_datasets/Annotations/"

train_folder="/home/yolo2/train"
valid_folder="/home/yolo2/valid"
test_folder="/home/yolo2/test"

labels=["RBC"]
architecture="Tiny Yolo"
input_size=416

max_box_per_img=20
anchors=[0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]

train_times=10
valid_times=1
nb_epochs=200
learning_rate=1e-5
batch_size=1
warmup_epochs=200
object_scale=5.0
no_object_scale=1.0
coord_scale=1.0
class_scale=1.0

imgs,lbs=parse_annotation(annotations_folder,images_folder,labels)      # parse annotation files into image path and labels dictionary

n_images=len(imgs)
print("Total number of images is {}".format(n_images))

np.random.shuffle(imgs)
print("Images shuffled randomly")

train_count=int(0.6*n_images)                          # train, valid, test split in the ratio 0.6 : 0.2 : 0.2
valid_count=int(0.2*n_images)
test_count=int(0.2*n_images)


train_imgs=imgs[:train_count]
val_imgs=imgs[train_count:train_count+valid_count]
test_imgs=imgs[train_count+valid_count:]

print("There are {} training images".format(train_count))
print("There are {} validation images".format(valid_count))
print("There are {} testing images".format(test_count))

train_list=[]
valid_list=[]
test_list=[]

if os.path.exists(train_folder):                        # check if train folder, valid folder, test folder exists else create them
    print("Train folder already exists")
else:
    os.mkdir("train")
    print("created train folder")

if os.path.exists(valid_folder):
    print("Valid folder already exists")
else:
    os.mkdir("valid")
    print("created valid folder")

if os.path.exists(test_folder):
    print("Test folder already exists")
else:
   os.mkdir("test")
   print("created test folder")

i=1
for img in train_imgs:
   train_list.append(img['filename'])
   pathh,d,ff,gg,ssd,hh,img_name=img['filename'].split("/")
   path=os.path.join("/home/yolo2/train/"+img_name)
#    print(path)
   img=cv2.imread(img['filename'])
   cv2.imwrite(path,img)
#    print("Train Image copied {} ".format(i))
   i+=1
print("Total number of train images copied is {}".format(i-1))

i=1
for img in val_imgs:
   valid_list.append(img['filename'])
   pathh,d,ff,gg,ssd,hh,img_name=img['filename'].split("/")
   path=os.path.join("/home/yolo2/valid/"+img_name)
#    print(path)
   img=cv2.imread(img['filename'])
   cv2.imwrite(path,img)
#    print(" Validation Image copied {} ".format(i))
   i+=1
print("Total number of validation images copied is {}".format(i-1))

i=1   
for img in test_imgs:
   test_list.append(img['filename'])
   pathh,d,ff,gg,ssd,hh,img_name=img['filename'].split("/") 
   path=os.path.join("/home/yolo2/test/"+img_name)
#    print(path)
   img=cv2.imread(img['filename'])
   cv2.imwrite(path,img)
#    print("Test Image copied {} ".format(i))
   i+=1
print("Total number of test images copied is {}".format(i-1))

yolo=YOLO(  architecture=architecture,                        # YOLO class from frontend.py
            input_size=input_size,                            # add gpus = 1,2,3 if gpu is used, for cpu ignore or put gpus=0 
            labels=labels,
            max_box_per_img=max_box_per_img,
            anchors=anchors,
            
        )
print("Model defined") 



yolo.train( train_imgs=train_imgs,                          # train class function from frontend.py
            valid_imgs=val_imgs,
            train_times=train_times,
            valid_times=valid_times,
            nb_epochs=nb_epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            warmup_epochs=warmup_epochs,
            object_scale=object_scale,
            no_object_scale=no_object_scale,
            coord_scale=coord_scale,
            class_scale=class_scale,
            saved_weights_name="best_weights.h5",
            train=True
         )


   



