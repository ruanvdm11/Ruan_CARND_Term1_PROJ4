# Advanced Lane Finding - Project 4

---

**Advanced Lane Finding Project**

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

[image1]: https://raw.githubusercontent.com/ruanvdm11/Ruan_CARND_Term1_PROJ4/master/Reference%20Images/CameraCalibration.png "Camera Calibration"
[image2]: https://raw.githubusercontent.com/ruanvdm11/Ruan_CARND_Term1_PROJ4/master/Reference%20Images/BeforeDistorted.png "Distorted"
[image3]: https://raw.githubusercontent.com/ruanvdm11/Ruan_CARND_Term1_PROJ4/master/Reference%20Images/Undistorted.png "Undistorted"
[image4]: https://raw.githubusercontent.com/ruanvdm11/Ruan_CARND_Term1_PROJ4/master/Reference%20Images/Unperspective.png "Unperspective"
[image5]: https://raw.githubusercontent.com/ruanvdm11/Ruan_CARND_Term1_PROJ4/master/Reference%20Images/Perpective.png "Perpective"
[image6]: https://raw.githubusercontent.com/ruanvdm11/Ruan_CARND_Term1_PROJ4/master/Reference%20Images/Transforms1.png "Transform Start"
[image7]: https://raw.githubusercontent.com/ruanvdm11/Ruan_CARND_Term1_PROJ4/master/Reference%20Images/Transforms3Normalize.png "Transform Normalise and HLS"
[image8]: https://raw.githubusercontent.com/ruanvdm11/Ruan_CARND_Term1_PROJ4/master/Reference%20Images/Transforms4ScaledSobel.png "Scaled Sobel"
[image9]: https://raw.githubusercontent.com/ruanvdm11/Ruan_CARND_Term1_PROJ4/master/Reference%20Images/Transforms2Sobel.png "Binary Sobel"
[image10]: https://raw.githubusercontent.com/ruanvdm11/Ruan_CARND_Term1_PROJ4/master/Reference%20Images/FindingPoints1Hist.png "Points Histogram"
[image11]: https://raw.githubusercontent.com/ruanvdm11/Ruan_CARND_Term1_PROJ4/master/Reference%20Images/FindingPoints2Windows.png "Sliding Windows"
[image12]: https://raw.githubusercontent.com/ruanvdm11/Ruan_CARND_Term1_PROJ4/master/Reference%20Images/FindingPoints3Poly.png "Polygon"
[image13]: https://raw.githubusercontent.com/ruanvdm11/Ruan_CARND_Term1_PROJ4/master/Reference%20Images/Final1Input.png "Final Input"
[image14]: https://raw.githubusercontent.com/ruanvdm11/Ruan_CARND_Term1_PROJ4/master/Reference%20Images/Final2Output.png "Final Output"
[image15]: https://raw.githubusercontent.com/ruanvdm11/Ruan_CARND_Term1_PROJ4/master/Reference%20Images/Video1_Screenshot.PNG "Video1"
[video1]: https://github.com/ruanvdm11/Ruan_CARND_Term1_PROJ4/blob/master/Reference%20Images/Video1.mp4 "Video"

### Here I will consider the rubric points ([Rubric](https://review.udacity.com/#!/rubrics/571/view) Points) individually and describe how I addressed each point in my implementation.  

---

## 1. Camera Calibration

### 1.1 Calibtration of the Camera That Was Used For Data Capture. 

The code for this step is contained in the cell 2 of the IPython notebook located `CARND_Term1_PROJ4.ipynb`.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objP` is just a replicated array of coordinates, and `objPoints` will be appended with a copy of it every time I successfully detect all chessboard corners in an image.  `imgPoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objPoints` and `imgPoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained the following results. An image is also shown of the chessboard corners that were detected.

![alt text][image1]

## 2. Pipeline (single images)

### 2.1 Correcting the Distortion of Each Image

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

Using the `mtx` and `dist` variables obtained from the camera calibration and the `cv2.undistort` function made it possible to account for camera distortion in each of the images. The result obtained is shown below:

![alt text][image3]

### 2.2 Creating a Perspective Image for 'Vertical' Line Identification.

The code for my perspective transform includes a function called `makePers()`, which appears in code cell 4 of `CARND_Term1_PROJ4.ipynb`.  The `makePers()` function takes [`An undistorted image`] as input. The `src` and `dist` points were hard coded into `makePers()`. The function is illustratted below

```python
def makePers(img):

    img_size = (img.shape[1], img.shape[0])
    src = np.float32(
    [[1100, 720],
     [690, 460],
     [590, 460],
     [200, 720]])
    
    dst = np.float32(
    [[1030, 680],
     [980, 0],
     [310, 0],
     [250, 680]])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped
```
The `src` and `dist` were obtained by using the `straightLines` that were supplied. Knowing that the lines were straight allows for the perspective to be calculated quite easily. The following image shows the original image as well as the images with perpective.

| Image for `src` and `dist`| Warped Image|
| :--| :--|
| ![alt text][image4]| ![alt text][image5]|


### 2.3 Transformation of Image to Binary Image
This section will be explained by the use of this start image:

![alt text][image6]

After the warped image had been calculated the `cv2.Sobel` function was implemented across the `x-axis`. This code was used to generate the binary image:

```python
def makeSobel(img):
    
    threshold = (90, 255)    
    dst = np.zeros_like(img)
    b=cv2.normalize(img,dst,0,255,cv2.NORM_MINMAX)    
    b = cv2.cvtColor(b, cv2.COLOR_RGB2HLS)    
    S = b[:,:,2]
    
    sobelx = cv2.Sobel(S, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(S, cv2.CV_64F, 0, 1)
    abssobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255*abssobelx/np.max(abssobelx))
    threshmin = 10
    threshmax = 255
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= threshmin) & (scaled_sobel <= threshmax)] = 1
    
    return sxbinary
```
The following shows the sequence of the transformation:
* 1) Convert 2 HLS and Normalise

![alt text][image7]
* 2) Scaled Sobel Output

![alt text][image8]
* 3) Binary Output

![alt text][image9]

As seen I only ustilised the sobel function in the x direction. However, I did convert the image to HLS in order to eliminate the effect of `L - Lightness` after a normalization of the image had been completed. The lightness array was used as a gray input to the sobel function.

In order to create a more robust approach it could be beneficial to use dual axis thresholds.



### 2.4 Finding the Polynomial to Represent the Lane Lines

* 1) The output binary image of the previous image was the starting point.
* 2) Next it was necessary to determine where there were large densities of non zero points. This was done by the help of the `np.histogram` function The next images shows the result for the bottom half of the image.

![alt text][image10]
* 3) However, it is necessary to evaluate the image at more than just two levels, therefor the images was divided into 9 windows to determine where the dense sections are. This is shown here:

![alt text][image11]

* 4) A point is then placed at the column where the largest density is for each window. These points are then input into the following code to generate the image thereafter:

```python

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
```
![alt text][image12]

* 5) Each array for the polygon lines are stored in order to obtain an average so that small inconsistencies in the images do not play such a big role. Also, The error for any new arrays are calculated and if the error threshold is exceeded the array is not added. The code can be seen below:

```python

leftData = np.zeros([1,720])
def leftLineAvg(leftLine):  ### This function saves each poly line array in order to determine an average (Left Side)

    global leftData
    newLeftLine = np.reshape(leftLine, ((1,720)))
    leftDataShape = np.shape(leftData)      
    if (len(leftData)>=2):    
        error = (leftData[leftDataShape[0]-1,:]-newLeftLine[0,:])/leftData[leftDataShape[0]-1,:]
        error = np.absolute(np.average(error))
        if (error)>0.18:
            print('left')
            print('Error: ',error)
            leftLine = leftData[leftDataShape[0]-1,:]
        else:
            leftData = np.vstack((leftData, leftLine))
    if len(leftData)==1:
        leftData = np.vstack((leftData, leftLine))
    if len(leftData)<20:
        summ = np.zeros((1,720))
        for i in range(0,len(leftData)):
            summ += leftData[i]
        leftavg = summ/(len(leftData)-1)
    if len(leftData)>=20:
        summ = np.zeros((1,720))
        for j in range(len(leftData)-20,len(leftData)):
            summ += leftData[j]
        leftavg = summ/(20)   
    return (leftavg.reshape((720,)))
```


### 2.5 Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The radaius of the curve calculation can be seen in code cell 4 of `CARND_Term1_PROJ4.ipynb`. The values obtained from the previous section was input into my `calcCurve()` function. 

The function to create the polygons is `np.polyfit`. This functions needs three inputs. Seeing as the order of the polynomial is 2 in this case the three inputs are:

* Vertical column values (y-axis)
* X values, each corresponding to a Y value
* 2 - which is the order ot the polynomial

Here is the code:
```python

curveData = np.zeros((1,1))
def calcCurve(leftx, rightx, ploty):

    global curveData
    ## Conversion factors
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    y_eval = np.max(ploty)
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5)np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Conversion to meters
    curveData = np.vstack((curveData, left_curverad))
    if len(curveData)>10:
        curveAvg = (sum(curveData[len(curveData)-10:len(curveData)]))/10
    else:
        curveAvg = sum(curveData[1:len(curveData)])/(len(curveData)-1)
    
    return curveAvg
```

The calculation of the vehicles relevant postions was done by assuming that the vehicle is always at the center of the images and subtraction that pixel value from the difference of the poly lines. Code is given below:

```python

def middleCalc(img, leftFit, rightFit):

    imshape = np.shape(img)
    carX = (imshape[1])/2
    laneX = leftFit[len(leftFit)-1]+(rightFit[len(rightFit)-1]-leftFit[len(leftFit)-1])/2
    midVal = (laneX-carX)*(3.7/700)
    return midVal
```


### 2.6 Final Output 'Unwarped' and Masked Onto Original Image

The code for this can be found in code cell 4 of `CARND_Term1_PROJ4.ipynb`. Here is a display of the final image showcasing the drivable area on the lane.

| Final Input Image| Final Output Image|
| :--| :--|
| ![alt text][image13]| ![alt text][image14]|


---

### Pipeline (video)

Here's a download [link to my video result](https://github.com/ruanvdm11/Ruan_CARND_Term1_PROJ4/blob/master/ProjectVideoResult.mp4)

Also, here is a link to how my model performs on the challenge video:
[link to Challenge 1 Result](https://github.com/ruanvdm11/Ruan_CARND_Term1_PROJ4/blob/master/Reference%20Images/Challenge_Video_1.mp4)


Here the video can be viewed:
[![alt text][image15]](https://www.youtube.com/watch?v=yNwpUtTncc4)

---

### Discussion

When I evaluated my model on the challenge videos I saw that there is a need to make the model more robust. One of the techniques I considered was to adjust the weighting in the sobel function.

Also, some image cleanup techniques can be implemented before the model is sent evaluated by the sobel function.

My first model did not look as smooth as it does now, because I was not applying an averaging filter to the polylines array. After the averaging had been implemented the performance increased significantly.

Also, it is important to keep calculation time as small as possible because in the ideal case this must be done in real time.
