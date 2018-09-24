import pickle
import cv2
import glob

# Read in the saved camera calibration result
dist_pickle = pickle.load( open("camera_cal/dist_pickle.p", "rb" ) )
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

# Make a list of images
images = glob.glob('test_images/*.jpg')

# performs image distortion correction and
# returns the undistorted image

def undistort(img, mtx1, dist1):
    undist = cv2.undistort(img, mtx1, dist1, None, mtx)
    return undist


for idx, fname in enumerate(images):
    # Read in an image
    img = cv2.imread(fname)
    undistorted = undistort(img, mtx, dist)
    cv2.imwrite('output_images/undistorted_'+str(idx+1)+'.jpg', undistorted)
    cv2.imshow('img', undistorted)
    cv2.waitKey(500)