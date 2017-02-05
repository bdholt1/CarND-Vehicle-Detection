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
            
            if clf.predict([test_img]):
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

def vote(accumulator, boxes):
    # Iterate through list of boxes
    for box in boxes:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        accumulator[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return accumulator

def multiscale_slide_window(image, clf):
    """
    Slide a window over the image at different scales.
    
    """
    
    window_image = np.copy(image)
    accumulator = np.zeros(image.shape[0:2])
    windows = []
    
    w = slide_window(image, clf, x_start_stop=[None, None], y_start_stop=[319, None], 
                        xy_window=(400, 400), xy_overlap=(0.9, 0.9))
    draw_boxes(window_image, w, color=(255, 0, 0), thick=6)
    vote(accumulator, w)
    windows.append(w)
    print("400x400 bounding boxes: ", len(w))
    
    w = slide_window(image, clf, x_start_stop=[20, 1260], y_start_stop=[339, 680], 
                        xy_window=(300, 300), xy_overlap=(0.9, 0.9))
    draw_boxes(window_image, w, color=(255, 255, 0), thick=6)
    vote(accumulator, w)
    windows.append(w)
    print("300x300 bounding boxes: ", len(w))    
    
    w = slide_window(image, clf, x_start_stop=[50, 1230], y_start_stop=[359, 640], 
                        xy_window=(200, 200), xy_overlap=(0.9, 0.9))
    draw_boxes(window_image, w, color=(0, 255, 255), thick=6)
    vote(accumulator, w)
    windows.append(w)
    print("200x200 bounding boxes: ", len(w))     
    
    w = slide_window(image, clf, x_start_stop=[100, 1180], y_start_stop=[379, 620], 
                        xy_window=(100, 100), xy_overlap=(0.8, 0.8))
    draw_boxes(window_image, w, color=(255, 0, 255), thick=6)
    vote(accumulator, w)
    windows.append(w)
    print("100x100 bounding boxes: ", len(w)) 
    
    w = slide_window(image, clf, x_start_stop=[250, 1030], y_start_stop=[379, 600], 
                        xy_window=(50, 50), xy_overlap=(0.6, 0.6))
    draw_boxes(window_image, w, color=(0, 255, 0), thick=6)
    vote(accumulator, w)
    windows.append(w)
    print("50x50 bounding boxes: ", len(w)) 
    
    return windows, window_image, accumulator

def label_detections(accumulator, threshold):
    accumulator_thresh = np.copy(accumulator)
    accumulator_thresh[accumulator <= threshold] = 0
    
    from scipy.ndimage.measurements import label
    labels = label(accumulator_thresh)
    
    return labels

def draw_labeled_bboxes(image, labels):
    labels_image = np.copy(image)
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(labels_image, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return labels_image

if __name__ == "__main__":

    pipeline = joblib.load('classifier.pkl')
    print(pipeline)

    image_files = glob.glob('./test_images/test*.jpg')

    fig, axes = plt.subplots(len(image_files), 4)

    for f,ax in zip(image_files, axes):
        image = cv2.imread(f)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        windows, window_image, accumulator = multiscale_slide_window(rgb, pipeline)
        
        labels = label_detections(accumulator, threshold=5)
        print(labels[1], 'cars found')

        labels_image = draw_labeled_bboxes(rgb, labels)

        ax[0].imshow(window_image)
        ax[1].imshow(accumulator, cmap='hot')
        ax[2].imshow(labels[0], cmap='gray')
        ax[3].imshow(labels_image)

    plt.show()