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

import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import os

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision.transforms import functional
from PIL import Image

# Versions of tensorflow used -------------------------
print("tf.__version__ is", tf.__version__)
print("tf.keras.__version__ is:", tf.keras.__version__)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


onnx_model = onnx.load("Material_Classifier.onnx")
k_model = onnx_to_keras(onnx_model, ['imageinput'])
keras.models.save_model(k_model, "Material_Classifier.h5", overwrite=True,include_optimizer=True)
# Versions of tensorflow used -------------------------
class Watcher:
    DIRECTORY_TO_WATCH = "update_here"

    def __init__(self):
        self.observer = Observer()

    def run(self):
        event_handler = Handler()
        self.observer.schedule(event_handler, os.getcwd(), recursive=True)
        self.observer.start()
        try:
            while True:
                time.sleep(5)
        except:
            self.observer.stop()
            print("Error")

        self.observer.join()


class Handler(FileSystemEventHandler):

    @staticmethod
    def on_any_event(event):
        if event.is_directory:
            return None

        elif event.event_type == 'created':
            # Take any action here when a file is first created.
            print("Received created event - %s." % event.src_path)
            model = tf.keras.models.load_model("Material_Classifier.h5")
            IMG_SIZE = 200
            count = 0
            time.sleep(10)
            start_time = time.time()
            array = []

            for file in os.listdir(r'C:\Users\Owen Lu\Desktop\Pytorch lessons\Keras\update_here'):
                path = os.path.join(r'C:\Users\Owen Lu\Desktop\Pytorch lessons\Keras\update_here',file)
                img_array = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                
                prediction_pic = new_array.reshape(-1,1,IMG_SIZE, IMG_SIZE)

                prediction = model.predict(prediction_pic)
                
                count += 1
                print(str(count))
            exec_time = time.time() - start_time
            print("Execution time in seconds: "  + str(exec_time))
            exit()


if __name__ == '__main__':
    w = Watcher()
    w.run()