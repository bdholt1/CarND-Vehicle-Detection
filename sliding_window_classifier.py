#!/usr/bin/env python

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob

import jsonpickle

from time import time

from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from feature_extraction import get_hog_features, HOGFeaturiser


# Define a function that takes an image,
# start and stop positions in both x and y, 
# window size (x and y dimensions),  
# and overlap fraction (for both x and y)
def slide_window(img, clf, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
                    
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    print("xspan = ", xspan, " yspan = ", yspan)
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0])) + 1
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1])) + 1
    print("nx_pix_per_step = ", nx_pix_per_step, " ny_pix_per_step = ", ny_pix_per_step)
    # Compute the number of windows in x/y
    nx_windows = np.int((xspan - xy_window[0]) / nx_pix_per_step) + 1
    ny_windows = np.int((yspan - xy_window[1]) / ny_pix_per_step) + 1
    print("nx_windows = ", nx_windows, " ny_windows = ", ny_windows)
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            
            # Append window position to list
            test_img = cv2.resize(gray[starty:endy, startx:endx], (64, 64))
            
            cv2.imshow('window', test_img)
            cv2.waitKey(1000)
            
            if clf.predict(test_img):
                window_list.append(((startx, starty), (endx, endy)))
                
    # Return the list of windows
    return window_list

# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(img, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    #return imcopy
    
def multiscale_slide_window(image, clf):
    """
    Slide a window over the image at different scales.
    
    """
    
    window_image = np.copy(image)    
    windows = []
    
    w = slide_window(image, clf, x_start_stop=[None, None], y_start_stop=[319, None], 
                        xy_window=(400, 400), xy_overlap=(0.9, 0.9))
    draw_boxes(window_image, w, color=(255, 0, 0), thick=6)
    windows.append(w)
    print("400x400 bounding boxes: ", len(w))
    
    w = slide_window(image, clf, x_start_stop=[20, 1260], y_start_stop=[339, 680], 
                        xy_window=(300, 300), xy_overlap=(0.9, 0.9))
    draw_boxes(window_image, w, color=(255, 255, 0), thick=6)
    windows.append(w)
    print("300x300 bounding boxes: ", len(w))    
    
    w = slide_window(image, clf, x_start_stop=[50, 1230], y_start_stop=[359, 640], 
                        xy_window=(200, 200), xy_overlap=(0.9, 0.9))
    draw_boxes(window_image, w, color=(0, 255, 255), thick=6)
    windows.append(w)
    print("200x200 bounding boxes: ", len(w))     
    
    w = slide_window(image, clf, x_start_stop=[100, 1180], y_start_stop=[379, 620], 
                        xy_window=(100, 100), xy_overlap=(0.8, 0.8))
    draw_boxes(window_image, w, color=(255, 0, 255), thick=6)
    windows.append(w)
    print("100x100 bounding boxes: ", len(w)) 
    
    w = slide_window(image, clf, x_start_stop=[250, 1030], y_start_stop=[379, 600], 
                        xy_window=(50, 50), xy_overlap=(0.6, 0.6))
    draw_boxes(window_image, w, color=(0, 255, 0), thick=6)
    windows.append(w)
    print("50x50 bounding boxes: ", len(w)) 
    
    return windows, window_image
    
if __name__ == "__main__":

    classifier_svm = joblib.load('classifier.pkl')
    pipeline = Pipeline([
        ('hog', HOGFeaturiser()),
        ('scaler', StandardScaler()),
        ('clf', classifier_svm),
    ])

    
    classifier = pipeline.set_params(hog__orientations=11,
                                     hog__pixels_per_cell=12,	
                                     hog__cells_per_block=1,
                                     clf__C=10,
                                     clf__gamma=0.001,
                                     clf__kernel='rbf')


    
    print(classifier)

    #with open('classifier.json', 'r') as f: 
    #   vec_repr = f.read()
    #    print(vec_repr)
    #    classifier = jsonpickle.decode(vec_repr)
    
    image_files = glob.glob('./test_images/*.jpg')

    for f in image_files:
        image = mpimg.imread(f)
        image = image.astype(np.float32)/255 # JPEGs are loaded at 0..255
        
        windows, window_image = multiscale_slide_window(image, classifier)
        
        # Plot the examples
        fig = plt.figure()
        plt.subplot(121)
        plt.imshow(image)
        plt.title('Example Image')
        plt.subplot(122)
        plt.imshow(window_image)
        plt.title('Detected vehicles')
        plt.show()
    