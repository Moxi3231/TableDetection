from matplotlib import colors
import numpy as np
import tensorflow as tf

from box_util import *
from random import choice, randint,random

from matplotlib.colors import cnames

from cv2 import (COLOR_BGR2GRAY,IMREAD_COLOR, COLOR_BGR2RGB, COLOR_GRAY2RGB, Canny, GaussianBlur, Sobel,
                 cvtColor, dilate, erode, imread, adaptiveThreshold, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY)

#MIN SIDE=500 #MAXSIDE + 575
def resize_pad_image(images, min_side=500.0, max_side=700.0, jitter=[450,550], stride=128.0,training_flag = False):
    #Change: images[0] [:2] If Three images were to be supplied 
    image_shape = tf.cast(tf.shape(images)[:2], dtype=tf.float32)

    if jitter is not None:
        min_side = tf.random.uniform((), jitter[0], jitter[1], dtype=tf.float32)
    
    ratio = min_side / tf.reduce_min(image_shape)
    
    if ratio * tf.reduce_max(image_shape) > max_side:
        ratio = max_side / tf.reduce_max(image_shape)
    image_shape = ratio * image_shape
    
    padded_image_shape = tf.cast(tf.math.ceil(
        image_shape / stride) * stride, dtype=tf.int32)
    #for image in images:
    image = tf.image.resize(images, tf.cast(image_shape, dtype=tf.int32))
    
    image = tf.image.pad_to_bounding_box(image, 0, 0, padded_image_shape[0], padded_image_shape[1])
    
    if training_flag and tf.random.uniform(()) > 0.5:
        image = tf.cast(image,dtype=tf.float32)
        g_noise = tf.random.uniform(
            image.shape, 
            minval = tf.random.uniform((),0.55,0.75,dtype=tf.float32), 
            maxval = 1.0, 
            dtype=tf.dtypes.float32, 
            seed=None)
        #tf.print("NOISE")
        image = image * g_noise
    #processed_images.append(image)
    return image, image_shape, ratio

def random_flip_image(image, boxes):
    #Change Here: image[0].shape[:2] for model with three image 
    image_shape = image.shape[:2]
    #tf.print("SHAPE",image_shape)
    
    boxes = tf.stack([boxes[..., 0] / image_shape[1], boxes[..., 1] / image_shape[0],
                    boxes[..., 2] / image_shape[1], boxes[..., 3] / image_shape[0]], axis=-1)
    
    if tf.random.uniform(()) > 0.5:
        #tf.print("IMAGE FLIP")
        image = tf.image.flip_left_right(image)
        boxes = tf.stack([1-boxes[:, 0]-boxes[:,2], boxes[:, 1], boxes[:, 2], boxes[:, 3]], axis=-1)
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_up_down(image)
        boxes = tf.stack([boxes[:,0], 1 - boxes[:,1] - boxes[:,3], boxes[:,2], boxes[:,3]], axis = -1)

    return image, boxes

def preprocess_image(images, image_labels,training_flag = False):
    #tf.print("LABELS",image_labels)
    
    bbox = tf.cast(image_labels[..., :4], dtype=tf.float32)
    class_ids = tf.cast(image_labels[..., 4], dtype=tf.int32)
    #tf.print("\n\n\n")
    
    #tf.print("Original Box",bbox)
    #tf.print("OShape",tf.shape(images))
    images, bbox = random_flip_image(images, bbox)
    #tf.print(bbox,tf.shape(images))
    images, image_shape, _ = resize_pad_image(images, training_flag = training_flag)

    #tf.print("\nNShape",image_shape)
    #tf.print("Normalized",bbox)
    bbox = tf.stack([
        bbox[:, 0] * image_shape[1],
        bbox[:, 1] * image_shape[0],
        bbox[:, 2] * image_shape[1],
        bbox[:, 3] * image_shape[0],
    ], axis=-1)
    #tf.print("New Box",bbox)
    bbox = convert_to_xy_center_from_xy_min_wh(bbox)
    #tf.print("Converted Box",bbox,"\n---------------")
    return images, bbox, class_ids

class ImageAugmenter:
    def __init__(self,training_flag = False):
        self.training_flag = training_flag
      

    @staticmethod
    def get_second_image(img):
        """
        Returns dilated image.
        """
        #imgb = (img <= 128)*255
        #imgb = imgb.astype(np.uint8)
        
        imgb = adaptiveThreshold(img, 255, ADAPTIVE_THRESH_MEAN_C,THRESH_BINARY, 15, -2)
        
        for _ in range(3):
            imgb = GaussianBlur(imgb,(5,5),-1)
            
        imgxy = Sobel(imgb,-1,dx=1,dy=1,ksize=5)
        
        imgxy = imgxy + imgb 

        for _ in range(3):
            imgxy = GaussianBlur(imgxy,(5,1),-1)

            imgxy = erode(imgxy,np.ones((3,5)),iterations = 2)
            imgxy = dilate(imgxy,np.ones((3,9)),iterations = 2)
            
            imgxy = ((imgxy > 15)*255).astype(np.uint8)
        return cvtColor(imgxy,COLOR_GRAY2RGB)

    @staticmethod
    def get_third_image(img):
        """
        Returns the image that only consist borders.
        """
        #imgb = (img <= 170)*255
        #imgb = imgb.astype(np.uint8)
        imgb = adaptiveThreshold(img, 255, ADAPTIVE_THRESH_MEAN_C,THRESH_BINARY, 15, -2)

        imgb1 = (img <= 125)*255
        imgb1 = imgb1.astype(np.uint8)
        
        imgb = (imgb & imgb1) | imgb1
        
        for _ in range(5):
            imgb = erode(imgb,np.ones((1,3)),iterations = 1)
            imgb = GaussianBlur(imgb,(7,7),-1)
            imgb = ((imgb!=0)*255).astype(np.uint8)
        
        imgb = dilate(imgb,np.ones((3,7)),iterations = 3)
        
        imgb = (Canny(imgb,170,255)>15)*255
        imgb = imgb.astype(np.uint8)

        return cvtColor(imgb,COLOR_GRAY2RGB)
        
    @staticmethod
    def convert_to_grayscale(img):
        return cvtColor(img, COLOR_BGR2GRAY)

    @staticmethod
    def read_image(img_path):
        return imread(img_path)

    @staticmethod
    def get_color():
        return choice(colors)

    def get_raw_image(self,imgpath:str):
        img = imread(imgpath,IMREAD_COLOR)
        return img

    def get_raw_thres_image(self,imgpath:str):
        img = self.get_raw_image(imgpath)
        img = self.convert_to_grayscale(img)
        img = adaptiveThreshold(img, 255, ADAPTIVE_THRESH_MEAN_C,THRESH_BINARY, 15, -2)
        return img
    
    def get_3d_image(self,img):
        return cvtColor(img,COLOR_GRAY2RGB)

    def get_thresholded_image(self,imgpath:str):
        img = self.get_raw_thres_image(imgpath)

        for _ in range(3):
            img = GaussianBlur(img,(9,9),-1)
        img = erode(img,kernel=np.ones((5,5)),iterations=3)
        img = dilate(img,kernel=np.ones((5,5)),iterations=1)
        return cvtColor(img,COLOR_GRAY2RGB)    

    def get_distotred_image(self,image):
        o_shape = image.shape
        im_shape = list(image.shape) 

        extender = randint(100,200)
        
        im_shape[0] += (2*extender)
        im_shape[1] += (2*extender)

        im_shape = tuple(im_shape)
        nimg = np.zeros(im_shape).astype(np.uint8)

        for channel in range(3):
            nimg[extender:extender + o_shape[0],extender:extender + o_shape[1],channel] = image[:,:,channel]

        return nimg,extender

    def get_all_3_images(self,imgpath:str):
        
        img = self.get_raw_image(imgpath)
        img_gray = self.convert_to_grayscale(img)
        im2 = self.get_second_image(img_gray)
        im3 = self.get_third_image(img_gray)

        return img,im2,im3
        
    