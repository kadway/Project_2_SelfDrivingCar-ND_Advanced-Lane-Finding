## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

[//]: # (Image References)

[image1]: ./camera_cal/calibration1.jpg "Original chessboard image"
[image2]: ./camera_chess_undist/chess_undist_calibration1.jpg "Undistorted chessboard image"
[image3]: ./test_images/straight_lines1.jpg "Test image"
[image4]: ./output_images/undistorted/straight_lines1.jpg "Undistorted test image"
[image5]: ./output_images/color_gradient_transformed/straight_lines1_combined_binary.jpg "Color and gradient transformed image"
[image6]: ./output_images/warped/_original_straight_lines1.jpg "Warped image test"
[image7]: ./output_images/fitted_poly/straight_lines1.jpg "Fitted poly"
[image8]: ./output_images/out_image/straight_lines1.jpg "Radius"
[image9]: ./radius_formula.png "Formula radius"
[video1]: ./output_videos/project_video.mp4 "Project video"

The Project
---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Camera Calibration

The code for this step is contained in the IPython notebook located in `./camera_calibration.ipynb`.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

Original Chessboard
![alt text][image1]

"Undistorted Chessboard"
![alt text][image2]

### Pipeline (single images)

#### 1. Example of a distortion-corrected image.

Here is an example of the `straight_lines1.jpg` original test image.

![alt text][image3]

The undistortion of an image is performed by the `undistort(img, img_path=0)` function in the `Advanced_lane_finding.ipynb` notebook.
Firstly the camera calibration parameters are loaded and then feed in to the openCV function `cv2.undistort(img, mtx, dist, None, mtx)` which returns the undistorted image.
The `img_path` is the image path and is only passed on to the function for the purpose of saving a copy of the modified image to the respective folder in `/output_images/`.

After the distortion correction, the test image looks like this:

![alt text][image4]

#### 2. Color transforms and gradients to create a thresholded binary image.
I used a combination of color and gradient thresholds to generate a binary image.
The thresholding steps are performed in function `transform(image, img_path=0)` in `Advanced_lane_finding.ipynb`.

The undistorted image was converted to gray scale and the OpenCV Sobel() function was used to apply x and y gradient to the image.
Sobel x and y gradients thresholded images. The x and y gradients were then thresholded into a binary image. (function `abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):`)

The gradient magnitude and direction were calculated from the Sobel x and y gradients. 

Gradient magnitude is calculated with: `np.sqrt(sobelx**2 + sobely**2)`
And gradient direction: `np.arctan2(np.absolute(sobely), np.absolute(sobelx))`
​
The color channels S and L were also thresholded and combined into the final binary image.
The image was first converted to HLS with `cv2.cvtColor(img, cv2.COLOR_RGB2HLS)` and then the individual color channels were thresholded.

The combined binary image is performed by combining all the binary images in the following way:
`((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)) | ((s_binary == 1) & (l_binary == 1))`

Here's an example of my output for this step using the `straight_lines1.jpg` test image.

![alt text][image5]

#### 3. Perspective transform.

The code for my perspective transform includes a function called `warper()`,  in the 4th code cell of the IPython notebook `Advanced_lane_finding.ipynb`.  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points. 
I chose to hardcode the source and destination points, according to the code provided in the writeup_template, in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image6]


#### 4. Identifying lane-line pixels and fitting their positions with a polynomial.

After applying a perspective transform, I created a histogram of the bottom half of the image.
The lane lines are more likely to be vertical close to the car, so after dividing the histogram in half, 
I use the maximum value of pixels from the left and right sides to decide where along the x-axis the lane lines are starting.
Knowing the starting point, I then applied the sliding window method to find the rest of the pixels belonging to the lane.
It consists in finding the non-zero pixels inside a predetermined window,
and after that re-centering the next window in the mean position of the pixels identified. 
The code for lane finding using this method is in the function `find_lane_pixels(binary_warped, fname=0)` in `Advanced_lane_finding.ipynb`.
After having the points of the lane line pixels, I fit a 2nd order polynomial to the left and right lanes.

The function `search_around_poly(binary_warped, fname=0)` is used when lanes were identified in a previous frame.
It searches for the lane pixels within a determined margin of the last fitted line and a when successful
the `class Line(): ` is used to store the last good left and right lane polynomial fits as well as the x and y pixels of each lane.

Here is an example of the fitted lane lines using the sliding window method:
![alt text][image7]

#### 5. Calculated radius of curvature of the lane and the position of the vehicle with respect to center.

The radius of curvature is done in function `measure_curvature_pixels(img_shape)`.
The relation is the following:   `ym_per_pix = 30/720 # meters per pixel in y dimension` and `xm_per_pix = 3.7/700 # meters per pixel in x dimension`.

And with the coeficients fom the new polynomial fit the radius is calculated with the following formula:

![alt text][image9]
 
This calculation is done with the code lines: 
for left lane `left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])`
and for the right lane `right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])`.

The left and right lane radius are then averaged and saved to `leftLane.radius_of_curvature` and `rightLane.radius_of_curvature`.

The distance of the car to the center of the lane is also calculated by doing the difference of the image center to the center of the lane (in meters): `offset = (img_shape[1]/2-(leftLane.bestx + rightLane.bestx)/ 2)*3.7/700`


#### 6. Example image of the result plotted back down onto the road with lane area clearly identified.

This step is implemented in `draw_lanes_text(warped_t_binary, undistorted_img, dst, src)`.
A blank image with the same shape of the binary warped image is used to draw a polygon identifing the lane lines.
The new image with the lanes drawn is then warped back into the original image space and then combined with the original image.
The radius and offset to lane center are then also drawn onto the original image.

![alt text][image8]

---

### Pipeline (video)

Here's a [link to my video result](./output_videos/project_video.mp4)

---

### Discussion

One big challenge in this project was to get a reasonable transformed image by applying color and gradient thresholds because there are many possible combinations. The combined result is just enough to be able to process the project_video. Shadows in an image are always hard to bypass, and also the sobel gradient threshold is sometimes bringing noise into the image. One way to clean the noise could be to apply a mask, however a mask should not be static so a smart way of applying a "cleaning mask" would have to be well thought. The used color channels used (S and L) where the ones that seemed better, but other color channels could be even better if the proper thresholds are found, this would be also an improvement that would require more exerimenting with more image frames besides the ones provided in test_images folder.

Besides the color and gradient transforms another point of failure for the pipeline is the inexistence of a proper sanity check.
After fitting the lines they should be checked for similar curvature, their horizontal distance and if they are roughly parallel.
This sanity checks would also help to decide when to search for pixels around the previous fittet polynomial or when to go back to sliding window method. The actual pipeline maybe be accepting wrong line fits and continuously searching for pixels around the worng fitted lines.



