#!/usr/bin/env python

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob

from pprint import pprint
from time import time
import logging

from skimage.feature import hog

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler


# Read in our vehicles and non-vehicles
notcars_files = glob.glob('features/non-vehicles/**/*.png')
print("Number of non-vehicles in dataset = ", len(notcars_files))

cars_files = glob.glob('features/vehicles/**/*.png')
print("Number of vehicles in dataset = ", len(cars_files))

# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    if vis == True:
        # Use skimage.hog() to get both features and a visualization
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block,cell_per_block), visualise=True, 
                                  transform_sqrt=True, feature_vector=False)
        return features, hog_image
    else:      
        # Use skimage.hog() to get features only
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block,cell_per_block), visualise=False, 
                                  transform_sqrt=True, feature_vector=feature_vec)
        return features

class HOGFeaturiser(BaseEstimator, TransformerMixin):
    def __init__(self, orientations=None, pixels_per_cell=None, cells_per_block=None):
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block

    def fit(self, X, y=None):
        """This method doesn't do anything. It exists purely for compatibility
        with the scikit-learn transformer API."""
        return self

    def transform(self, X):
        """Given a list of images, return a list of HOG feature vectors."""
        features = []
        for img in X:
            vec = get_hog_features(img, self.orientations, self.pixels_per_cell, 
            				      self.cells_per_block, vis=False, feature_vec=True)
            features.append(vec)
        arr = np.array(features)
        print("shape of HOG features = ", arr.shape)
        return arr

pipeline = Pipeline([
    ('hog', HOGFeaturiser()),
    ('scaler', StandardScaler()),
    ('clf', SVC()),
])

# uncommenting more parameters will give better exploring power but will
# increase processing time in a combinatorial way
parameters = {
    'hog__orientations': (7, 9, 11),
    'hog__pixels_per_cell': (4, 6, 8, 10),
    'hog__cells_per_block': (2, 3, 4),
    'clf__C': (1)
}

if __name__ == "__main__":
    # multiprocessing requires the fork to happen in a __main__ protected
    # block

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

    cars_images = []
    for f in cars_files:
        image = mpimg.imread(f)
        cars_images.append(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))
        flip_lr = cv2.flip(image, 1)
        cars_images.append(cv2.cvtColor(flip_lr, cv2.COLOR_RGB2GRAY))
    cars_labels = np.ones(len(cars_images))
    
    notcars_images = []
    for f in notcars_files:
        image = mpimg.imread(f)
        notcars_images.append(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))
        flip_lr = cv2.flip(image, 1)
        notcars_images.append(cv2.cvtColor(flip_lr, cv2.COLOR_RGB2GRAY))
    notcars_labels = np.zeros(len(notcars_images))
    
    data = cars_images + notcars_images
    print("total dataset size = ", len(data))
    labels = np.concatenate((cars_labels, notcars_labels))
    print("total labels size = ", labels.shape)
    pprint(labels)

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.10, random_state=42)

    # find the best parameters for both the feature extraction and the classifier
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=6, verbose=1)

    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    pprint(parameters)
    t0 = time()
    grid_search.fit(X_train, y_train)
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

    print("Best score on test: %0.3f" % grid_search.best_estimator_.score(X_test, y_test))
    
    print(grid_search.cv_results_)


        





