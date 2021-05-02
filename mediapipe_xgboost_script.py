import numpy
import cv2
import os
import pickle
from tqdm import tqdm
from sklearn.model_selection import train_test_split

x_test = numpy.load("./x_test_mp.npy")
y_test = numpy.load("./y_test_mp.npy")

loaded_model = pickle.load(open("./mediapipe_xgboost.sav", 'rb'))

print(loaded_model.score(x_test, y_test))
