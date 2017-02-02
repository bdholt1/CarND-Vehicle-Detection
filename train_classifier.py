#!/usr/bin/python


import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob

import jsonpickle

import pandas as pd

from pprint import pprint
from time import time
import logging

from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.preprocessing import StandardScaler

from feature_extraction import get_hog_features, HOGFeaturiser

# Read in our vehicles and non-vehicles
notcars_files = glob.glob('features/non-vehicles/**/*.png')
print("Number of non-vehicles in dataset = ", len(notcars_files))

cars_files = glob.glob('features/vehicles/**/*.png')
print("Number of vehicles in dataset = ", len(cars_files))


def load_images(image_files):
    images = []
    for f in image_files:
        image = mpimg.imread(f)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        images.append(gray)
        flip_lr = cv2.flip(gray, 1)
        images.append(flip_lr)
    return images

cars_images = load_images(cars_files)
cars_labels = np.ones(len(cars_images))

notcars_images = load_images(notcars_files)
notcars_labels = np.zeros(len(notcars_images))

data = cars_images + notcars_images
print("total dataset size = ", len(data))
labels = np.concatenate((cars_labels, notcars_labels))
print("total labels size = ", labels.shape)
pprint(labels)


pipeline = Pipeline([
    ('hog', HOGFeaturiser()),
    ('scaler', StandardScaler()),
    ('clf', SVC()),
])

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.95, random_state=42)

classifier = pipeline.set_params(hog__orientations=11,
                                 hog__pixels_per_cell=12,	
                                 hog__cells_per_block=1,
                                 clf__C=10,
                                 clf__gamma=0.001,
                                 clf__kernel='rbf').fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
print("Test score: %0.3f" % accuracy_score(y_pred, y_test))


print(pipeline.named_steps['clf'])
joblib.dump(pipeline.named_steps['clf'], 'classifier.pkl') 

#vec_repr = jsonpickle.encode(classifier)
#print(vec_repr)

#with open('classifier.json', 'w') as f: 
#    f.write(vec_repr) 
