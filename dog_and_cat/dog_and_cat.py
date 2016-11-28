import os

import glob



from keras.models import Sequential

from keras.layers.core import Flatten, Dense, Dropout

from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D

from keras.optimizers import SGD

import cv2, numpy as np



def VGG_16(weights_path=None):

    model = Sequential()

    model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))

    model.add(Convolution2D(64, 3, 3, activation=’relu’))

    model.add(ZeroPadding2D((1,1)))

    model.add(Convolution2D(64, 3, 3, activation=’relu’))

    model.add(MaxPooling2D((2,2), strides=(2,2)))



    model.add(ZeroPadding2D((1,1)))

    model.add(Convolution2D(128, 3, 3, activation=’relu’))

    model.add(ZeroPadding2D((1,1)))

    model.add(Convolution2D(128, 3, 3, activation=’relu’))

    model.add(MaxPooling2D((2,2), strides=(2,2)))



    model.add(ZeroPadding2D((1,1)))

    model.add(Convolution2D(256, 3, 3, activation=’relu’))

    model.add(ZeroPadding2D((1,1)))

    model.add(Convolution2D(256, 3, 3, activation=’relu’))

    model.add(ZeroPadding2D((1,1)))

    model.add(Convolution2D(256, 3, 3, activation=’relu’))

    model.add(MaxPooling2D((2,2), strides=(2,2)))



    model.add(ZeroPadding2D((1,1)))

    model.add(Convolution2D(512, 3, 3, activation=’relu’))

    model.add(ZeroPadding2D((1,1)))

    model.add(Convolution2D(512, 3, 3, activation=’relu’))

    model.add(ZeroPadding2D((1,1)))

    model.add(Convolution2D(512, 3, 3, activation=’relu’))

    model.add(MaxPooling2D((2,2), strides=(2,2)))



    model.add(ZeroPadding2D((1,1)))

    model.add(Convolution2D(512, 3, 3, activation=’relu’))

    model.add(ZeroPadding2D((1,1)))

    model.add(Convolution2D(512, 3, 3, activation=’relu’))

    model.add(ZeroPadding2D((1,1)))

    model.add(Convolution2D(512, 3, 3, activation=’relu’))

    model.add(MaxPooling2D((2,2), strides=(2,2)))



    model.add(Flatten())

    model.add(Dense(4096, activation=’relu’))

    model.add(Dropout(0.5))

    model.add(Dense(4096, activation=’relu’))

    model.add(Dropout(0.5))

    model.add(Dense(1000, activation=’softmax’))



    if weights_path:

        model.load_weights(weights_path)



    return model



if name == “main“:



    # Test pretrained model

    model = VGG_16(‘vgg16_weights.h5’)

    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(optimizer=sgd, loss=’categorical_crossentropy’)



    dogs=[251, 268, 256, 253, 255, 254, 257, 159, 211, 210, 212, 214, 213, 216, 215, 219, 220, 221, 217, 218, 207, 209, 206, 205, 208, 193, 202, 194, 191, 204, 187, 203, 185, 192, 183, 199, 195, 181, 184, 201, 186, 200, 182, 188, 189, 190, 197, 196, 198, 179, 180, 177, 178, 175, 163, 174, 176, 160, 162, 161, 164, 168, 173, 170, 169, 165, 166, 167, 172, 171, 264, 263, 266, 265, 267, 262, 246, 242, 243, 248, 247, 229, 233, 234, 228, 231, 232, 230, 227, 226, 235, 225, 224, 223, 222, 236, 252, 237, 250, 249, 241, 239, 238, 240, 244, 245, 259, 261, 260, 258, 154, 153, 158, 152, 155, 151, 157, 156]



    cats=[281,282,283,284,285,286,287]



    path = os.path.join(‘imgs’, ‘test’, ‘*.jpg’)

    files = glob.glob(path)

    result=[]



    for fl in files:

        flbase = os.path.basename(fl)



        im = cv2.resize(cv2.imread(fl), (224, 224)).astype(np.float32)

        im[:,:,0] -= 103.939

        im[:,:,1] -= 116.779

        im[:,:,2] -= 123.68

        im = im.transpose((2,0,1))

        im = np.expand_dims(im, axis=0)



        out = model.predict(im)

        p = np.sum(out[0,dogs]) / (np.sum(out[0,dogs]) + np.sum(out[0,cats]))

        result.append((flbase,p))



    result=sorted(result, key=lambda x:x[1], reverse=True)

    for x in result:

        #print x[0],x[1]

        print x[0]
