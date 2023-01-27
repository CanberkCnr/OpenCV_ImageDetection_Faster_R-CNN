#LIBRARIES
import torchvision
from torchvision import  transforms 
import torch
from torch import no_grad

#Libraries for gettin data from the web
import requests

#Libraries for image processing and visualization
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def get_predictions(pred,threshold=0.8,objects=None ):
    """
    This function will assign a string name to a predicted class and eliminate predictions whose likelihood  is under a threshold 
    
    pred: a list where each element contains a tuple that corresponds to information about  the different objects; Each element includes a tuple with the class yhat, probability of belonging to that class and the coordinates of the bounding box corresponding to the object 
    image : frozen surface
    predicted_classes: a list where each element contains a tuple that corresponds to information about  the different objects; Each element includes a tuple with the class name, probability of belonging to that class and the coordinates of the bounding box corresponding to the object 
    thre
    """


    predicted_classes= [(COCO_INSTANCE_CATEGORY_NAMES[i],p,[(int(box[0]), int(box[1])), (int(box[2]), int(box[3]))]) for i,p,box in zip(list(pred[0]['labels'].numpy()),pred[0]['scores'].detach().numpy(),list(pred[0]['boxes'].detach().numpy()))]
    predicted_classes=[  stuff  for stuff in predicted_classes  if stuff[1]>threshold ]
    
    if objects  and predicted_classes :
        predicted_classes=[ (name, p, box) for name, p, box in predicted_classes if name in  objects ]
    return predicted_classes

#Draws box around each object

def draw_box(predicted_classes,image,rect_th= 10,text_size= 3,text_th=3):
    """
    draws box around each object 
    
    predicted_classes: a list where each element contains a tuple that corresponds to information about  the different objects; Each element includes a tuple with the class name, probability of belonging to that class and the coordinates of the bounding box corresponding to the object 
    image : frozen surface 
   
    """
    img=(np.clip(cv2.cvtColor(np.clip(image.numpy().transpose((1, 2, 0)),0,1), cv2.COLOR_RGB2BGR),0,1)*255).astype(np.uint8).copy()
    for predicted_class in predicted_classes:
   
        label=predicted_class[0]
        probability=predicted_class[1]
        box=predicted_class[2]
 
        cv2.rectangle(img, box[0], box[1],(0, 255, 0), rect_th) # Draw Rectangle with the coordinates
        cv2.putText(img,label, box[0],  cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th) 
        cv2.putText(img,label+": "+str(round(probability,2)), box[0],  cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    del(img)
    del(image)

#Free up some memory
def save_RAM(image_=False):
    global image, img, pred
    torch.cuda.empty_cache()
    del(img)
    del(pred)
    if image_:
        image.close()
        del(image)
        
#Load Pre-Trained Faster R-CNN (from COCO(Common Object in Context))
model_ = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model_.eval()

for name, param in model_.named_parameters():
    param.requires_grad = False
print("done")

#Save RAM
def model(x):
    with torch.no_grad():
        yhat = model_(x)
    return yhat

#Here your 91 Classes for Image Detection
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
len(COCO_INSTANCE_CATEGORY_NAMES)

#Object Localization
img_path='jeff_hinton.png'
half = 0.5
image = Image.open(img_path)

image.resize( [int(half * s) for s in image.size] )

plt.imshow(image)
plt.show()

#Create a transform object to conver the image to a TENSOR
transform = transforms.Compose([transforms.ToTensor()])
img = transform(image) #Convert the image to a Tensor

#Make a prediction
pred = model([img])#IF you call model_([img]) directly but it will use more RAM

pred[0]['labels'] # 35 different class for prediction
#Ordered by likelihood scores for potential objects

pred[0]['scores'] #Likelihood(ihtimal) for each class

#The class number corresponds to the index of the list with the corresponding category name
index=pred[0]['labels'][0].item()
COCO_INSTANCE_CATEGORY_NAMES[index]
#Output: Person

#we have the coordinates of the Bounding Box
bounding_box=pred[0]['boxes'][0].tolist()
bounding_box

#These components correspond to the top-left corner and bottom-right corner of the rectange,more precisely:

t,l,r,b=[round(x) for x in bounding_box] # Top(t), left(l), bottom(b), right(r)

#Convert the tensor to an open CV array and plot an image with the box
img_plot=(np.clip(cv2.cvtColor(np.clip(img.numpy().transpose((1, 2, 0)),0,1), cv2.COLOR_RGB2BGR),0,1)*255).astype(np.uint8)
cv2.rectangle(img_plot,(t,l),(r,b),(0, 255, 0), 10) # Draw Rectangle with the coordinates
plt.imshow(cv2.cvtColor(img_plot, cv2.COLOR_BGR2RGB))
plt.show()
del img_plot, t, l, r, b

#Localize Object(using get_predictions function)
#The input is the predictions pred and the objects
pred_class=get_predictions(pred,objects="person")
draw_box(pred_class, img)
del pred_class

#Set a threshold
get_predictions(pred,threshold=1,objects="person")

#no output as the likelihood is not 100%, try a threshold of 0.98 and use the function draw_box
pred_thresh=get_predictions(pred,threshold=0.98,objects="person")
draw_box(pred_thresh,img)
del pred_thresh

#Delete Object to save Memory
save_RAM(image_=True)

#Multiple Objects
img_path='DLguys.jpeg'
image = Image.open(img_path)
image.resize([int(half * s) for s in image.size])
plt.imshow(np.array(image))
plt.show()

#Set a Threshold to detect the object, 0.8 to work
img = transform(image)
pred = model([img])
pred_thresh=get_predictions(pred,threshold=0.8,)
draw_box(pred_thresh,img,rect_th= 1,text_size= 0.5,text_th=1)
del pred_thresh

#Or use the object parameters
pred_obj=get_predictions(pred,objects="person")
draw_box(pred_obj,img,rect_th= 1,text_size= 0.5,text_th=1)
del pred_obj

save_RAM(image_=True)

#Object Detection
img_path='istockphoto-187786732-612x612.jpeg'
image = Image.open(img_path)
image.resize( [int(half * s) for s in image.size] )
plt.imshow(np.array(image))
plt.show()
del img_path

#Set a threshold, detech all objects whose likelihood is above that threshold(EŞİK).
img = transform(image)
pred = model([img])
pred_thresh=get_predictions(pred,threshold=0.97)
draw_box(pred_thresh,img,rect_th= 1,text_size= 1,text_th=1)
del pred_thresh

#WE CAN SPECIFY THE OBJECTS WE WOULD LIKE TO CLASSIFY,FOR EXAMPLE, CATS AND DOG
# img = transform(image)
# pred = model([img])
# pred_obj=get_predictions(pred,objects=["dog","cat"])
# draw_box(pred_obj,img,rect_th= 1,text_size= 0.5,text_th=1)
# del pred_obj

#If we set the threshold too low, we may detect objects with a low likelihood of being correct; here, we set the threshold to 0.7, and we incorrectly  detect a cat 
# pred_thresh=get_predictions(pred,threshold=0.70,objects=["dog","cat"])
# draw_box(pred_thresh,img,rect_th= 1,text_size= 1,text_th=1)
# del pred_thresh

save_RAM(image_=True)

#Other Objects
img_path='watts_photos2758112663727581126637_b5d4d192d4_b.jpeg'
image = Image.open(img_path)
image.resize( [int(half * s) for s in image.size] )
plt.imshow(np.array(image))
plt.show()
del img_path

#Detect
img = transform(image)
pred = model([img])
pred_thresh=get_predictions(pred,threshold=0.997)
draw_box(pred_thresh,img)
del pred_thresh

save_RAM(image_=True)

#Update photo with URL
url='https://www.plastform.ca/wp-content/themes/plastform/images/slider-image-2.jpg'

image = Image.open(requests.get(url, stream=True).raw).convert('RGB')
del url

img = transform(image )
pred = model([img])
pred_thresh=get_predictions(pred,threshold=0.95)
draw_box(pred_thresh, img)
del pred_thresh

save_RAM(image_=True)
