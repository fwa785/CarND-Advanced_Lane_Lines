# Advanced Lane Lines Project

---

**Overview**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted Chessboard"
[image2]: ./examples/undistorted_test_image.png "Undistorted Test Image"
[image3]: ./examples/test3_binary_combo.jpg "Binary Example"
[image4]: ./examples/straight_lines1_warped.jpg "Warp Example1"
[image5]: ./examples/test3_warped.jpg "Warp Example2"
[image6]: ./examples/test2_color_fit_lines.jpg "Fit Visual"
[image7]: ./examples/test3_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

###Code 

The code of the project is here[https://github.com/fwa785/CarND-Advanced_Lane_Lines/]

### Camera Calibration

First the camera is calibrated.

#### Object points and Image points
The code for this step is contained in camera_calibrate.py. 

I start by preparing "object points", which will be the (x, y, z) coordinates of 
the chessboard corners in the world. Here I am assuming the chessboard is fixed 
on the (x, y) plane at z=0, such that the object points are the same for each 
calibration image.  Thus, `objp` is just a replicated array of coordinates, 
and `objpoints` will be appended with a copy of it every time I successfully detect 
all chessboard corners in a calibration image under camera_cal/directory.  `imgpoints` 
will be appended with the (x, y) pixel position of each of the corners in the image 
plane with each successful chessboard detection.  

The `imagepoints` and the `objpoints` are saved in a file named wide_dist_pickle.p.

#### Undistort the Image

The next step is to use the `objpoints` and `imgpoints` to compute the camera 
calibration and distortion coefficients and undistort the image. The code for this step is 
contained in undistort_img.py.

The `objpoints` and `imgpoints` are loaded from the file. Next using the `cv2.calibrateCamera()` 
function I found the camera matrix and the distortion efficients. Then I applied this distortion 
correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

The pipeline for a single image is implemented in pipeline.py. I will describe in details for
each pipeline step.

#### 1. Undistort an image

The first step of the pipeline is to undistort an image. Using the functions provided in undistort_img.py,
I applied undistortion to each test image provided under test_images directory.

Here is an example of the original image and undistorted sample image.

![alt text][image2]

#### 2. Gradient and Color Thresholded Binary Image
The second step of the pipeline is to use gradient and color threshold to generate binary image.
 
The gradient.py file provides multiple functions to select the pixels of an image based on 
gradient in the threshold range.

The color_sel.py file provides functions to select the pixels of an image based on the color 
value in the threshold range.

After a few experiment, I decided to use 
* kernel size 7 for sobel gradient in both x and y 
* threshold 20 to 255 for both x and y gradient
* OR threshold 100 to 255 for saturation in the HLS color space
* OR threshold 20 to 100 for hue in the HLS color space
 
The example shows a thresholded binary image with the above parameters. As you can see, the lanes 
are cleared selected.

![alt text][image3]

#### 3. Perspective Transformation

The code for my perspective transform is in transform_img.py. It provided a function
warp_transform() to take the input image and some other parameters and transfer the image
to a bird view image.

It selects a trapezoid region in the source image, and transforms the region into a recetangle.

The trapezoid region is selected by the input parameters to the warp_transform() function:
* upper_w: the percentage of original image's width for the upper width of the trapeziod
* lower_w: the percentage of original image's width for the lower width of the trapeziod
* upper_h: the percentage of the original image's height for y coordinate of the upper edge of the trapeziod
* lower_h: the percentage of the original image's height for y coordinate of the lower edge of the trapeziod

The size of the recetangle region is controlled by the original image size of offset on x coordinate.
Because we're trying to transform the image and looking at the lanes from birdview, and the lanes are more
vertical from birdview, we don't have to leave some margin on y coordinate. But the lanes can have curvature,
it is better for us to leave some margins on the x coordinate from the lanes to the edge of the destination 
image so the curves don't go off the destination image.

The source vertices and the destination vertices are generated by the following python code:

```python
src = np.float32([[img_width * (0.5 - upper_w / 2), img_height * upper_h],
                  [img_width * (0.5 + upper_w / 2), img_height * upper_h],
                  [img_width * (0.5 + lower_w / 2), img_height * lower_h],
                  [img_width * (0.5 - lower_w / 2), img_height * lower_h]])

dest = np.float32([[ img_width* offset, 0],
                   [ img_width* (1-offset), 0],
                   [ img_width* (1-offset), img_height],
                   [ img_width* offset, img_height]])
```

After some experiment, I select the following parameters:

* upper_w = 0.08
* lower_w = 0.80
* upper_h = 0.62
* lower_h = 0.95
* offset = 0.20

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 589, 446      | 256, 0        | 
| 691, 446      | 576, 0        |
| 1152, 684     | 576, 720      |
| 1152, 684     | 256, 720      |

I verified that my perspective transform was working as expected by checking the warped image for
straight lane lines as vertical like shown below.

![alt text][image4]

I also check the warped image for a curved line and make sure the lane lines are parallel:

![alt text][image5]

#### 4. Identify lane-line pixels and Polynomial Fitting

I use the sliding window algorithm to identify the lane lane pixels. The code is implemented in
lane_tracker.py file.

I first take a histogram along all the columns in the lower half of the image like this:

```python
histogram = np.sum(img[img.shape[0]//2:,:], axis=0)
```

Then the bases of the left and right lane are found by looking for the maximum histogram value on the
left half and right half of the image.

```python

midpoint = np.int(histogram.shape[0]//2)
leftx_base = np.argmax(histogram[:midpoint])
rightx_base = np.argmax(histogram[midpoint:]) + midpoint

```

Next I define the window to search the lanes from the base. Each iteration, the window is sliding 
upward, and the search range on x coordinate is adjusted based on the current found lane position on 
x coordinate and the window width margin as shown below.

```python

for window in range(nwindow):
    win_y_low = img.shape[0] - (window + 1) * window_height
    win_y_high = img.shape[0] - window * window_height
    win_xleft_low = leftx_current - margin
    win_xleft_high = leftx_current + margin
    win_xright_low = rightx_current - margin
    win_xright_high = rightx_current + margin

```

The x and y position of the nonzero pixels are identified first:

```python

# Identify the x and y positions of all nonzero pixels
nonzero = img.nonzero()
nonzeroy = np.array(nonzero[0])
nonzerox = np.array(nonzero[1])

```

Then the indices for the nonzero pixels in the current seaching window is found as below:

```python

# identify the indicices for the non-zero pixels in the window
good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                  (nonzerox >= win_xleft_low) &
                  (nonzerox <  win_xleft_high)).nonzero()[0]

good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                  (nonzerox >= win_xright_low) &
                  (nonzerox <  win_xright_high)).nonzero()[0]

left_lane_inds.append(good_left_inds)
right_lane_inds.append(good_right_inds)

```

And the new lane center on the x coordinate will be updated based on the mean of the nonzero pixels'
position: 

```python

if(len(good_left_inds) > minpix):
    leftx_current = np.int(np.mean(nonzerox[good_left_inds]))

if (len(good_right_inds) > minpix):
    rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

```

After searching through window by window in the y direction, I extract the x and y positions of
the lanes pixels

```python

# Extract left and right line pixel positions
leftx = nonzerox[left_lane_inds]
lefty = nonzeroy[left_lane_inds]
rightx = nonzerox[right_lane_inds]
righty = nonzeroy[right_lane_inds]

```

The x and y positions for left lane and right lane are fit in a second order polynomial function 
as below:

```python

# Fit a second order polynomial to each
left_fit = np.polyfit(lefty, leftx, 2)
right_fit = np.polyfit(righty, rightx, 2)

```

The fitting curve is draw on the wrapped image. The picture below shows the left lane pixels in red, right lane pixels 
in blue, the searching window in green box and the polynomial fitting curve in yellow:

![alt text][image6]


#### 5. Curvature of the Lane 

The curvature of the lanes are calculated in the following steps:

* Define the conversions in x and y from pixels space to real world unit meters
```python
ym_per_pix = 40 / 720  # meters per pixel in y dimension
xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
```
* Fit the x and y in real world units
```python
# fit in the world space
left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
```
* Calculate the curvature 
```python
# calculate the radius of curvature
left_curverad = ((1+((2*left_fit_cr[0]*car_y+left_fit_cr[1])**2))**1.5)/np.absolute(2*left_fit_cr[0])
right_curverad = ((1+((2*right_fit_cr[0]*car_y+right_fit_cr[1])**2))**1.5)/np.absolute(2*right_fit_cr[0])
```
#### 6. the Position of the Vehicle

Assuming the camera is mounted at the center of the car. The car's y position is at the bottom of the image,
and x position is at the center of the image.

```python
# the car's position in world space
car_y = lane_found_img.shape[0]*ym_per_pix
car_x = (lane_found_img.shape[1]*xm_per_pix)/2.
```

Then calculate the left lane's x position and right lane's x position, and the center of the lane is in the middle
of the two lanes.

```python

# calculate the left x and right x using the world space fitting
left_fitx_lane = left_fit_cr[0]*(car_y**2) + left_fit_cr[1]*car_y + left_fit_cr[2]
right_fitx_lane = right_fit_cr[0]*(car_y**2) + right_fit_cr[1]*car_y + right_fit_cr[2]

# calculate the position of the center of the lanes
lane_center = (left_fitx_lane + right_fitx_lane)/2.

```
The difference of the center of the lane and the car's x position of the car's position from the lane center.

```python
# calculate the car's position offset from the center of the lane
car_offset = (lane_center - car_x)
```
#### 7. Mark Lane Area

A list of y positions is generated by np.linspace function, then I used the polynomial fitting function to 
generated the left lane's x positions and right lane's x positions by giving y positions. Each pair of (x, y) is a point in the image, with a list 
of (x, y) points, I filled the polygon shape with green color using function cv2.fillPoly().

After the lane area is marked on the transformed space, now I used the perspective transformation to transform
the marked lane area back to the original undistorted image's space, and added the marked area to the original 
undistorted image. The result is shown as below, with the radius and the car position printed as well.

![alt text][image7]

---

### Pipeline (video)

#### Video Output

Using the same pipeline for single image, I applied it to the project_video, and generated an output video with
the marked lane area, with the radius of the lane and the car position printed. 

Here's a [link to my video result](./output_videos/project_video.mp4)

---

### Discussion

#### Problems

On the project video, occassionally, the edge of the divider is identified as the left lane, and the marked area 
is extended to the divider. That is because sometimes the lights or the color or shadow cause of lane less visible
than the edge of the divider. To fix that, I use region of interest to select the interested region with a polygon 
and filter out the edge of the divider. With that change, the output for the project video doesn't mistakely identify
the edge of the divider as the left lane.

The pipeline doesn't work with the challenge video. There is a crack on the ground. The gradient threshold selection will
pick it up. But a good color threshold selection should filter it out. That means my color selection function still needs
to be tuned to work better with the lane selection.

The pipeline also doesn't work with the harder challenge video. The hard challenge video has a big curve. I think the 
problems can be at:
* The perspective transform needs to be tuned to get more accurate parallel lanes
* The sliding window algorithm be tuned to select the lane pixels well