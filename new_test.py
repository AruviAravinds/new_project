# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 13:19:12 2022

@author: Aravind
"""

#Reports parameters for each batch (total 1096) for each epoch.
#For 10 epochs we should see 10960

#################################################

#Test trained model on a few images...
from matplotlib import pyplot
from numpy import vstack
from os import listdir
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from numpy import asarray, load
from numpy.random import randint
from keras.preprocessing.image import load_img
model = load_model('model_010960.h5')

# plot source, generated and target images
def plot_images(src_img, gen_img, tar_img):
	images = vstack((src_img, gen_img, tar_img))
	# scale from [-1,1] to [0,1]
	images = (images + 1) / 2.0
	titles = ['Source', 'Generated', 'Expected']
	# plot images row by row
	for i in range(len(images)):
		# define subplot
		pyplot.subplot(1, 3, 1 + i)
		# turn off axis
		pyplot.axis('off')
		# plot raw pixel data
		pyplot.imshow(images[i])
		# show title
		pyplot.title(titles[i])
	pyplot.show()


def load_images(path, size=(256,512)):
	src_list, tar_list = list(), list()
	# enumerate filenames in directory, assume all are images
	for filename in listdir(path):
		# load and resize the image
		pixels = load_img(path + filename, target_size=size)
		# convert to numpy array
		pixels = img_to_array(pixels)
		# split into satellite and map
		sat_img, map_img = pixels[:, :256], pixels[:, 256:]
		src_list.append(sat_img)
		tar_list.append(map_img)
	return [asarray(src_list), asarray(tar_list)]

def preprocess_data(data):
	# load compressed arrays
	# unpack arrays
	X1, X2 = data[0], data[1]
	# scale from [0,255] to [-1,1]
	X1 = (X1 - 127.5) / 127.5
	X2 = (X2 - 127.5) / 127.5
	return [X1, X2]

# dataset path
path = 'C:/maps/val/'
# load dataset
[src_images, tar_images] = load_images(path)
print('Loaded: ', src_images.shape, tar_images.shape)

data = [src_images, tar_images]
dataset = preprocess_data(data)
[X1,X2]=dataset
ix = randint(1, len(X1)+1,1)
print ("The value of ix is" , (ix))
src_image, tar_image = X1[ix], X2[ix]
# generate image from source
gen_image = model.predict(src_image)
# plot all three images
plot_images(src_image, gen_image, tar_image)