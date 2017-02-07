##Writeup

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./data_examples.png
[image2]: ./hog_visualisation.png
[image3]: ./sliding_windows.png
[image4]: ./sliding_window_classifier_test_images.png
[image5]: ./accumulator_labelling.png
[image6]: ./video_accumulator.png
[image7]: ./video_labels.png
[image7]: ./video_boxes.png


###Writeup / README

###Histogram of Oriented Gradients (HOG)

####1. Extracting HOG features.

The code for this step is contained in lines 1 through 45 of the file called `feature_extraction.py`, where I created a
scikit-learn transform that accepts a list of images and computes the HOG features for those images based on the provided
parameters.  The 3 parameters that are tunable here are the number of orientation bins, the number of pixels per cell
and the number of cells per block.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes

![Data examples][image1]


From the Dalal Triggs 2005 paper I got the sense that the color space doesn't make much difference to performance, so I focused my effort on exploring the various parameters to the HOG feature. After reading in images, I converted them to grayscale and flipped them horizontally to duplicate the dataset.

The dataset contains 17936 non-vehicle images and 17944 vehicle images.

Here is an example using grayscale and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![HOG Visualisation][image2]

####2. Choosing HOG parameters.

Because the HOGFeaturiser is written as a scikit-learn transform, it can be used within a pipeline to do a grid search over all the possible parameter combinations.
I then explored different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`)
considering the orientations (5, 7, 9, 11, 13), pixels_per_cell (4, 6, 8, 10, 12) and cells_per_block (1, 2, 3, 4), 300 combinations in total. See `feature_analysis.py` for the full details.

I used 5% of the dataset to do the grid search with a linear SVM classifer and found that the best parameters were `orientations=9`, `cells_per_block=(1, 1)`, `pixels_per_cell=(12, 12)`.
With this configuration the best cross-validation score on the 5% training set was 97.1%, and this generalised well to the remaining 95% of hold-out data by scoring 97.0%

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


Then I tested various SVM configurations with a grid search and found that an SVM with a non-linear RBF kernel, with penalty C=10 and gamma=0.001 performed best.

Based on this analysis, i.e. on 5% of the training data we can achieve over 97% accuracy on the 95% test data, it was clear to me
that color data would not add a significant performance improvement.

####3. Training a classifier.

From the results of the previous step, I trained a non-linear SVM with RBF kernel on 95% of the data (see `train_classifier.py`) with a score of 99.2% accuracy on the test data.

###Sliding Window Search

####1. Multiscale sliding window.

I decided to search at 5 different scales, from 400x400 at the largest, through 300x300, 200x200, 100x100 and 50x50.  At each stage I reduce the region over which to do the detections based on the prior probability of seeing a vehicle in that location at the given scale. I took special care to reduce the overlap at smaller windows so that the total number of windows to classify remained manageable (a maximum of low hundreds).

Here are some examples of the sliding windows, using colors to differentiate them: red=400x400, yellow=300x300, cyan=200x200, purple=100x100 and green=50x50.  I chose these image sizes because they corresponded to the the largest scale I would expect to classify at (i.e. when the car is nearest to the camera) and the smallest image that can be realistically be classified (when the car is furthest).  Then I chose 3 image sizes between, adjusting the overlap ratio to smaller for the small boxes to contain the total number of windows that need to be classified.

![sliding windows][image3]


####2. Example detections

Applying my pipeline to these sliding windows resulted in good detections with very few false positives. In fact, it detected some cars in the oncoming lane that I struggled to detect. Here are some example images:

![sliding window detections][image4]

####2. Example labelling

The next step was to do hough voting into an accumulator for each of the detections and find the peaks.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the accumulator.

![labels][image5]

---

### Video Implementation

####1. Pipeline
My pipeline performs reasonably well, there are some short occurances of false positives and the bounding box size jumps around a bit, bit it's clearly tracking the vehicles.
Here's a [link to my video result](./project_video_output.mp4)


####2. Filtering false positives

I recorded the positions of positive detections in each frame of the video and voted that detection into an accumulator. The most recent `N=5` accumulators are kept and summed to yield an accumulator which I thresholded (`threshold=20`) to identify vehicle positions. I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

#### Here are six frames and their corresponding accumulator images:

![accumulators][image6]

#### Here is the output of `scipy.ndimage.measurements.label()` on the integrated accumulator from all six frames:
![labels][image7]

#### Here the resulting bounding boxes are drawn onto the last frame in the series:
![bounding boxes][image8]



---

### Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

This was a classic implementation of the Dalal Triggs SVM+HOG detector and I decided to stick with this configuration to see where the traditional approach falls flat.  I was extremely surprised how well the SVM with RBF kernel was able to classify the HOG patches, and what was most interesting about the grid search was that it chose a low-dimensional vector (11 orientations, 12 pixels_per_cell and 1 cell per block = 275 dimensions) rather than the higher dimensional possibilities over my grid search space. Clearly the RBF SVM was able to find an effective decision boundary in a higher dimensional projection.  The downside here was that the RBF SVM is is slow to run, approximately 0.4 FPS. I did a profile using `cProfile` and found that about 60% of that was taken by the SVM classifier, 30% by computing HOG features for each patch and 10% for the rest.  I spent a bit of time trying to compute the HOG descriptor over the entire image but found it challenging to work that into the pipeline. I also tried replacing the RBF SVM with a linear SVM (and best HOG parameters of 11 orientations, 4 pixels_per_cell and 2 cells per block) but the classifier was only operating at 94% and that yielded too many false positives.  One approach to solve this would be hard negative mining which I would have done if I were going to persue this project further.

Another area where my detector fails is that it's unable to distinguish between 2 separate vehicles and treats them as a single detection when they are too close or occlude each other.  An approach to resolving this is to use a filter (like a Kalman filter) to track the centroid of each detection box in image coordinates.

If I was going to pursue this project further I would implement semantic segmentation to classify each pixel into a class (tree, road, lane, car, sky etc) and use that as a prior on detections and to filter out false positives.  I would try out alternative classifiers e.g. Random Forests with short decision paths (by limiting max_depth) that are likely to be able to classify the image patches much faster.  There have been many recent improvements to tracking and detection, a brief look at some recent CVPR/ECCV and ICCV papers have provided a few insights on starting suggestions.


