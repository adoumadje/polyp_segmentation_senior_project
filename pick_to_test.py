from math import fabs
import tkinter as tk 
from tkinter import filedialog
import cv2
import numpy as np 
from keras.models import load_model
import os 
from plot import plot_picked_prediction


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


NETWORK_SIZE = (128,128)
TRESHOLD = 0.5

# Load model
MODEL_PATH = ".\\Models\\polyp_segmentation_bs=16_ep=1024_model.hdf5"
model = load_model(MODEL_PATH, compile=False)

root = tk.Tk()
root.withdraw()

file_path = filedialog.askopenfilename()

img = cv2.imread(file_path, cv2.IMREAD_COLOR)
img_heigth, img_width, _ = img.shape
ORIGINAL_SIZE = (img_width, img_heigth)
test_img = np.array(img)

# Normalize data 
test_img = test_img / 255
test_img_input = cv2.resize(test_img, NETWORK_SIZE)
test_img_input = np.expand_dims(test_img_input, 0)

prediction = (model.predict(test_img_input)[0,:,:,0] > TRESHOLD).astype(np.uint8)
prediction = cv2.resize(prediction, ORIGINAL_SIZE)

# show images
plot_picked_prediction(test_img,prediction)



# input("press any key to exit...")
