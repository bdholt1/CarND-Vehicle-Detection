##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in lines 1 through 45 of the file called `feature_extraction.py`, where I created a
scikit-learn transform that accepts a list of images and computes the HOG features for those images based on the provided
parameters.  The 3 parameters that are tunable here are the number of orientation bins, the number of pixels per cell
and the number of cells per block.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes

![alt text][data_examples.png]


From the Dalal Triggs 2005 paper I got the sense that the color space doesn't make much difference to performance, so I focused my effort
on exploring the various parameters to the HOG feature. After reading in images, I converted them to grayscale and flipped them horizontally
to duplicate the dataset.


The dataset contains 17936 non-vehicle images and 17944 vehicle images.



Here is an example using the `RGB` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

####2. Explain how you settled on your final choice of HOG parameters.

Because the HOGFeaturiser is written as a scikit-learn transform, it can be used within a pipeline to do a grid search over all the possible parameter combinations.
I then explored different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`)
considering the orientations (5, 7, 9, 11, 13), pixels_per_cell (4, 6, 8, 10, 12) and cells_per_block (1, 2, 3, 4). See `feature_analysis.py` for the full details.

I used 5% of the dataset to do the grid search with a linear SVM classifer and found that the best parameters were 1 cell_per_block, 11 orientation bins and 12 pixels_per_cell.
With this configuration the best cross-validation score on the 5% training set was 97.1%, and this generalised well to the remaining 95% of hold-out data
by scoring 97.0%

The detailed results were displayed in a Pandas dataframe, an extract of which is shown here

    mean_fit_time  mean_score_time  mean_test_score  mean_train_score param_hog__cells_per_block param_hog__orientations param_hog__pixels_per_cell  \
0        9.264974         4.767070         0.957770          0.998029                          1                       5                          4   
1        4.313093         2.173311         0.960023          0.998029                          1                       5                          6   
2        3.219087         1.659255         0.964527          0.996059                          1                       5                          8   
3        2.337224         1.129000         0.960023          0.994932                          1                       5                         10   
4        1.955618         0.964115         0.962275          0.992962                          1                       5                         12   
5       11.617766         5.108875         0.951577          0.997748                          1                       7                          4   
6        4.954706         2.399121         0.959459          0.997748                          1                       7                          6   
7        3.828630         1.702304         0.966779          0.997185                          1                       7                          8   
8        2.426752         1.192216         0.965090          0.996340                          1                       7                         10   
9        1.985604         0.994176         0.966779          0.995777                          1                       7                         12   
10      13.111449         6.365714         0.957770          0.998311                          1                       9                          4   
11       5.708373         2.901169         0.961712          0.998311                          1                       9                          6   
12       4.246415         1.952229         0.967905          0.997466                          1                       9                          8   
13       2.894939         1.262941         0.966779          0.997185                          1                       9                         10   
14       2.259227         1.083486         0.969032          0.996059                          1                       9                         12   
15      14.645743         6.381905         0.949887          0.998874                          1                      11                          4   
16       6.559427         2.860898         0.955518          0.998311                          1                      11                          6   
17       4.704322         2.074652         0.970158          0.997185                          1                      11                          8   
18       2.861834         1.293629         0.966779          0.997748                          1                      11                         10   
19       2.214648         1.102223         0.971284          0.996059                          1                      11                         12   
20      16.557534         7.064403         0.951014          0.998874                          1                      13                          4   
21       8.268868         3.377706         0.959459          0.998029                          1                      13                          6   
22       4.623072         2.452690         0.968468          0.997185                          1                      13                          8   
23       3.096842         1.384241         0.964527          0.997185                          1                      13                         10   
24       2.413026         1.168423         0.969595          0.996059                          1                      13                         12   
25      17.536168         7.320724         0.952140          0.998029                          2                       5                          4   
26       6.328422         3.327637         0.958896          0.997466                          2                       5                          6   
27       4.579805         2.169629         0.960586          0.995777                          2                       5                          8   
28       2.836059         1.399701         0.958333          0.994088                          2                       5                         10   
29       1.959382         0.905891         0.959459          0.990991                          2                       5                         12 


Then I tested various SVM configurations and found that an SVM with a non-linear RBF kernel, with penalty C=10 and gamma=0.001 performed best.

Based on this analysis, that on 5% of the training data we can achieve over 97% accuracy on the 95% test data, it was clear to me
that color data would not add a significant performance improvement.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

From the results of the previous step, I trained a non-linear SVM with RBF kernel on 95% of the data (see `train_classifier.py`) with a
score of 99.2% accuracy on the test data.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to try to minimize false positives and reliably detect cars?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

