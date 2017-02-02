#!/usr/bin/env python

import numpy as np

from skimage.feature import hog

from sklearn.base import BaseEstimator, TransformerMixin

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
    def __init__(self, orientations=9, pixels_per_cell=8, cells_per_block=2):
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
