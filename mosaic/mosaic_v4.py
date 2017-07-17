# coding: utf-8
# 鉴黄模型
from keras.models import model_from_json
from keras.preprocessing.image import array_to_img, img_to_array, load_img
import h5py
import os
from PIL import Image
from scipy.misc import imread, imresize, imsave
import numpy as np

print ('Begin')


from keras.applications import vgg16
from keras import backend as K
from keras.models import Model,Sequential
from keras.layers import Flatten, Dense, Input
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import decode_predictions, preprocess_input


img_width = 224
img_height = 224
batch_size = 32
if K.image_dim_ordering() == 'th':
    img_size = (3, img_width, img_height)
else:
    img_size = (img_width, img_height, 3)
mosaic = Input(batch_shape=(None,) + img_size)
model = vgg16.VGG16(input_tensor=mosaic,weights='imagenet', include_top=False)

for l in model.layers:
    l.trainable=False
#model.summary()

def preprocess_image(image_path):
    path = image_path.split(' ')[0]
    y = [int(image_path.split(' ')[1])]
    img = load_img(path, target_size=(img_width, img_height))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg16.preprocess_input(img)
    y = np.array(y)
    return img, y

def predict_image(image_path):
    img = load_img(image_path, target_size=(img_width, img_height))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg16.preprocess_input(img)
    return img

def deprocess_image(x):
    if K.image_dim_ordering() == 'th':
        x = x.reshape((3, img_width, img_height))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((img_width, img_height, 3))
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def preprocess_data(path, labs):
    data = []
    label = []
    filelist = os.listdir(path)
    for f in filelist:
        if np.random.random()>0.1:
	    continue
	preimage = preprocess_image(path + f)
        if preimage <> 'error':
            #print type(preimage)
            data.extend(preimage)
            label.extend([labs])
    return data, label

def generate_arrays_from_file(path):
    while 1:
        f = open(path)
        for line in f:
            # create numpy arrays of input data
            # and labels, from each line in the file
            img, y = preprocess_image(line)
            yield (img, y)
        f.close()


final_model = Sequential()
final_model.add(model)
final_model.add(Flatten(name='flatten'))
final_model.add(Dense(2048, activation='relu', name='fc1'))
final_model.add(Dense(2048, activation='relu', name='fc2'))
final_model.add(Dense(1, activation='sigmoid', name='predictions'))

final_model.compile(optimizer='Adadelta', loss='binary_crossentropy', metrics=['accuracy'])
#model.get_weights()[0][0][0]


#final_model.fit(data, label, batch_size=16, nb_epoch=1,shuffle=True,verbose=1,validation_split=0.05)
path = '/data/zhaihuixin/data/train_head.txt'
predictPath = '/data/zhaihuixin/data/train_tail_3000.txt'

final_model.fit_generator(generate_arrays_from_file(path),
    samples_per_epoch=128, 
    max_q_size=8,
    nb_worker=4,
    pickle_safe=True,
    nb_epoch=2000,
    nb_val_samples=256,
    validation_data=generate_arrays_from_file(predictPath))



final_model.save_weights('/data/zhaihuixin/src/jianhuang.h5')

print ('Try to Predict input images:')
print ('=====================================')

re_img = predict_image('/data/zhaihuixin/6.jpg')  # this is a jpg image (size (1, 64, 64, 3))
print final_model.predict(re_img)
print final_model.predict_proba(re_img)

path = '/data/zhaihuixin/gol075/'
files = os.listdir(path) 
for f in files:
    x_img = predict_image(path+f)
    print (f + ' probability is: %3.3f%% ' %(100*final_model.predict_proba(x_img)))



