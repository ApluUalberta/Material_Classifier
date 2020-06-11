import os

# Give the GPU permission to allocate memory to NN execution
os.environ['TF_KERAS'] = '1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# Give the GPU permission to allocate memory to NN execution


import onnx
import keras
import onnx2keras
from onnx2keras import onnx_to_keras


import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers

import keras2onnx

from tensorflow.python.keras.models import load_model
from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread

import cv2
from tqdm import tqdm
import random
import pickle


import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision.transforms import functional
from PIL import Image

# Versions of tensorflow used -------------------------
print("tf.__version__ is", tf.__version__)
print("tf.keras.__version__ is:", tf.keras.__version__)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
# Versions of tensorflow used -------------------------


# transform the given images --------------------------
PIL_transform = transforms.ToPILImage()
tensor_transform = transforms.ToTensor()
# transform the given images --------------------------

# Loading the onnx model ------------------------------
onnx_model = onnx.load("Material_Classifier.onnx")
k_model = onnx_to_keras(onnx_model, ['imageinput'])
keras.models.save_model(k_model, "Material_Classifier.h5", overwrite=True,include_optimizer=True)
# Loading the onnx model ------------------------------


# Folder and image size initialization ----------------
IMG_SIZE=200
CARBON = "CARBON/"
FIBERGLASS = "FIBERGLASS/"
LABELS = {CARBON: 0, FIBERGLASS: 1}
# Folder and image size initialization ----------------

# Counts and training data ----------------------------
training_data = []
carbon_count = 0
fiberglass_count = 0
# Counts and training data ----------------------------

# Make Training data ---------------------------------------------
for label in LABELS:
    for f in tqdm(os.listdir(label)):
        if "jpg" in f:
            try:
                path = os.path.join(label,f)
                img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))

                # randomize numbers ----------------------------------------------------------------------------------
                #degrees = abs(random.uniform(0.0,10.0)) # Get the degree range
                #scale = random.uniform(.5,2.0) # Get the scale range
                #horizontal_translation_factor = random.uniform(0.0,.2) # Get the translation factor range
                #vertical_translation_factor = random.uniform(0.0,.2)
                # randomize numbers ---------------------------------------------------------------------------------- 

                # transform -----------------------------
                #PIL_image = PIL_transform(img)
                #PIL_image = transforms.functional.affine(PIL_image,degrees,(horizontal_translation_factor, vertical_translation_factor),scale,shear = 0, resample=0,fillcolor=None)
                # if label == self.NEGATIVE:
                    # save = PIL_image.save("Transforms/Bad/" + f)

                # elif label == self.POSITIVE:
                    # save = PIL_image.save("Transforms/Good/" + f)
                
                #tensor_transform(PIL_image)
                # transform -----------------------------

                # we append the data --------------------
                training_data.append([np.array(img), np.eye(2)[LABELS[label]]])
                # we append the data --------------------

                # increment --------------------------------
                if label == CARBON:

                    carbon_count =carbon_count + 1
                elif label == FIBERGLASS:

                    fiberglass_count = fiberglass_count + 1
                # increment --------------------------------


            except Exception as e:
                print("bad things")
                pass

    np.random.shuffle(training_data)
    print('carbon:', carbon_count)
    print('fiberglass:', fiberglass_count)
print("training data length:", len(training_data))
# Make Training data ---------------------------------------------

# We add regularizers to every layer of the NN -------------------------------------
k_model.get_layer(index=0).kernel_regularizer = regularizers.l2(0.0001)
k_model.get_layer(index=0).bias_regularizer = regularizers.l2(0.0001)

k_model.get_layer(index=1).kernel_regularizer = regularizers.l2(0.0001)
k_model.get_layer(index=1).bias_regularizer = regularizers.l2(0.0001)

k_model.get_layer(index=2).kernel_regularizer = regularizers.l2(0.0001)
k_model.get_layer(index=2).bias_regularizer = regularizers.l2(0.0001)

k_model.get_layer(index=3).kernel_regularizer = regularizers.l2(0.0001)
k_model.get_layer(index=3).bias_regularizer = regularizers.l2(0.0001)

k_model.get_layer(index=4).kernel_regularizer = regularizers.l2(0.0001)
k_model.get_layer(index=4).bias_regularizer = regularizers.l2(0.0001)

k_model.get_layer(index=5).kernel_regularizer = regularizers.l2(0.0001)
k_model.get_layer(index=5).bias_regularizer = regularizers.l2(0.0001)

k_model.get_layer(index=6).kernel_regularizer = regularizers.l2(0.0001)
k_model.get_layer(index=6).bias_regularizer = regularizers.l2(0.0001)
# We add regularizers to every layer of the NN -------------------------------------

# Batch separation and saving into files to be loaded back in later ----------------
X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)
    
X = np.array(X).reshape(-1,1,IMG_SIZE,IMG_SIZE, 1)
y = np.array(y)

pickle_out = open("X.pickle", "wb")
pickle.dump(X,pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y,pickle_out)
pickle_out.close()
# Batch separation and saving into files to be loaded back in later ----------------

# Test and Training separation -----------------------------------------------------
X = pickle.load(open("X.pickle","rb"))
y = pickle.load(open("y.pickle","rb"))

VAL_PCT = 0.1  # lets reserve 10% of our data for validation
val_size = int(len(X)*VAL_PCT)

train_X = X[:-val_size]
train_y = y[:-val_size]

test_X = X[-val_size:]
test_y = y[-val_size:]
# Test and Training separation -----------------------------------------------------

# Compile our given model ----------------------------------------------------------
k_model.compile(loss="mean_squared_error", optimizer="adam",metrics=['accuracy'])
# Compile our given model ----------------------------------------------------------

# Train ----------------------------------------------------------------------------
#k_model.fit(train_X,train_y, batch_size=1, epochs=100)
# Train ----------------------------------------------------------------------------

# Test -----------------------------------------------------------------------------
k_model.evaluate(test_X, test_y, batch_size=1)
k_model.evaluate(train_X, train_y, batch_size=1)
# Test -----------------------------------------------------------------------------