import tensorflow as tf
import numpy as np
# from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
# import tensorflow.contrib.slim as slim
import os
import scipy.misc
import scipy
import gzip
import pickle
import time
np_rng = np.random.RandomState(42)

def PreprocessImages(images, dh, dw, ht, wt, nshuffles = 0, doreshape=1):
    #resize
    if doreshape==1:
        h = w = int( np.sqrt(images.shape[1])) #45
        images = np.reshape(images, [images.shape[0],h,w])
    else:
        h=w=images.shape[1]
#     print(h,w)
    
    
    #shift data left, right, and diagonally
    shift_h = h - dh#32
    shift_w = w - dw#32
    shift_h = int( np.floor( (h - dh)/2 ) )
    shift_w = int( np.floor( (w - dw)/2 ) )
    shifted_images = np.copy(images[:,:dh,:dw])  #np.copy(images[:,:32,:32])
#     print(shifted_images.shape)
    for i in range(shift_w):
        for j in range(shift_h):
    #         print(images[:, i:h-i , j:w-j][:,:32,:32].shape)
#            shifted_images = np.vstack((shifted_images, images[:, i:h-i , j:w-j][:,:32,:32] ))
            stack_images = images[:, i:h-i , j:w-j][:,:dh,:dw]
#             print(stack_images.shape)
            shifted_images = np.vstack((shifted_images, stack_images ))
    images = shifted_images

    #add data with swapped axis order (x, y, xy)
    images2 = images[:, ::-1, ::]
    images3 = images[:, ::, ::-1]
    images4 = images[:, ::-1, ::-1]
    images = np.concatenate((images,images2,images3,images4))


    #add data with swapped axes (up-down)
    # images2 = np.swapaxes(images,1,2)
    # images = np.concatenate((images,images2))

    #truncate  
    #ht=wt=32
    data_x = images[:,:h,:w,None]
    print("Data shape: ",data_x.shape)
    print("Original Data min, max:", data_x.min(), data_x.max() )

    #transform
    data_x = data_x - data_x.min()
    data_x = data_x / data_x.max()
    print("Scaled, Shifted Data min, max:", data_x.min(), data_x.max() )

    # #log-transform
    # data_x = np.exp(data_x)
    # print("Log-Transformed, Scaled, Shifted Data min, max:", data_x.min(), data_x.max() )

    #center
    # data_x = (data_x - np.mean(data_x,axis=0))*2
    data_x = (data_x - 0.5)*2
    print("Centered, Log-Transformed, Scaled, Shifted Data min, max:", data_x.min(), data_x.max() )
    print("Data shape: ", data_x.shape)


    #shuffle the order of the data
    #nshuffles = 10
    for i in range(nshuffles):
        data_x = shuffle(data_x)

    #show examples
    # sample_directory = './figsTut' #Directory to save sample images from generator in.
    # save_images(np.reshape(data_x[0:100],[100,32,32]),[10,10],sample_directory+'/fig'+'_examples_'+'.png')
    return data_x

def shuffles( data, nshuffles = 10):
    for i in range(nshuffles):
        data = shuffle(data)
    return data


def gray_plot(im,d=1,d1=1,new_figure=True):
    if new_figure:
        plt.figure(figsize=[d,d1])
    plt.imshow(im,interpolation='None',cmap='Greys')
    
def show_examples(x,square=True,h=9,w=9):
    N = x.shape[0]
    if square:
        d = int(np.ceil(np.sqrt(N)))
        d1 = int(np.ceil(N/d))
    else:
        d = N
        d1 = 1
        
    im = np.ones([1+d1*(h+1),1+d*(w+1)])
    for i in range(d1):
        for j in range(d):
            c = i*d + j
            if c<N:
                im[1+i*(h+1):(i+1)*(h+1),1+j*(w+1):(j+1)*(w+1)] = x[c,:].reshape([h,w])
    gray_plot(im,d,d1,True)


    #This function performns a leaky relu activation, which is needed for the discriminator network.
def lrelu(x, leak=0.2, name="lrelu"):
     with tf.variable_scope(name):
         f1 = 0.5 * (1 + leak)
         f2 = 0.5 * (1 - leak)
         return f1 * x + f2 * abs(x)
    
#The below functions are taken from carpdem20's implementation https://github.com/carpedm20/DCGAN-tensorflow
#They allow for saving sample images from the generator to follow progress
def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))

def inverse_transform(images):
    return (images+1.)/2.

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1]))

    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w] = image

    return img


from sklearn import utils as skutils
def list_shuffle(*data):
    idxs = np_rng.permutation(np.arange(len(data[0])))
    if len(data) == 1:
        return [data[0][idx] for idx in idxs]
    else:
        return [[d[idx] for idx in idxs] for d in data]

def shuffle(*arrays, **options):
    if isinstance(arrays[0][0], str):
        return list_shuffle(*arrays)
    else:
        return skutils.shuffle(*arrays, random_state=np_rng)









