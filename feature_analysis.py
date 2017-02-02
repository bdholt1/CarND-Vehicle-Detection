#!/usr/bin/env python

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob

import pandas as pd

from pprint import pprint
from time import time
import logging

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

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

def visualise_data():
    # Generate a random index to look at a car image
    ind = np.random.randint(0, len(cars_files))
    # Read in the image
    image = mpimg.imread(cars_files[ind])
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Define HOG parameters
    orient = 9
    pix_per_cell = 8
    cell_per_block = 2
    # Call our function with vis=True to see an image output
    features, hog_image = get_hog_features(gray, orient,
                            pix_per_cell, cell_per_block,
                            vis=True, feature_vec=False)

    # Plot the examples
    fig = plt.figure()
    plt.subplot(121)
    plt.imshow(image, cmap='gray')
    plt.title('Example Car Image')
    plt.subplot(122)
    plt.imshow(hog_image, cmap='gray')
    plt.title('HOG Visualization')
    plt.show()


def grid_search_params(X_train, X_test, y_train, y_test, pipeline, parameters):

    # find the best parameters for both the feature extraction
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)

    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    pprint(parameters)
    t0 = time()
    grid_search.fit(X_train, y_train)
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set: ",  grid_search.best_estimator_.get_params())
    print("Best score on test: %0.3f" % grid_search.best_estimator_.score(X_test, y_test))

    pd.set_option('display.width', 200)
    df = pd.DataFrame(grid_search.cv_results_)
    print(df)

if __name__ == "__main__":
    # multiprocessing requires the fork to happen in a __main__ protected
    # block

    visualise_data()

    cars_images = load_images(cars_files)
    cars_labels = np.ones(len(cars_images))

    notcars_images = load_images(notcars_files)
    notcars_labels = np.zeros(len(notcars_images))

    data = cars_images + notcars_images
    print("total dataset size = ", len(data))
    labels = np.concatenate((cars_labels, notcars_labels))
    print("total labels size = ", labels.shape)
    pprint(labels)

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.95, random_state=42)

    pipeline = Pipeline([
        ('hog', HOGFeaturiser()),
        ('scaler', StandardScaler()),
        ('clf', SVC()),
    ])

    # uncommenting more parameters will give better exploring power but will
    # increase processing time in a combinatorial way
    hog_parameters = {
        'hog__orientations': (5, 7, 9, 11, 13),
        'hog__pixels_per_cell': (4, 6, 8, 10, 12),
        'hog__cells_per_block': (1, 2, 3, 4),
    }

    best_hog_params = grid_search_params(X_train, X_test, y_train, y_test, pipeline, hog_parameters)

    svm_param_grid = [
        {'clf__C': [1, 10, 100, 1000], 'clf__kernel': ['linear']},
        {'clf__C': [1, 10, 100, 1000], 'clf__gamma': [0.001, 0.0001], 'clf__kernel': ['rbf']},
    ]

    best_svm_params = grid_search_params(X_train, X_test, y_train, y_test, pipeline, svm_param_grid)




