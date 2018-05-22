import cv2
import pickle

def load_cal_dist():
    # load the pickle file
    file = open("wide_dist_pickle.p", "rb")
    dist_pickle = pickle.load(file)

    objpoints = dist_pickle["objpoints"]
    imgpoints = dist_pickle["imgpoints"]

    return objpoints, imgpoints

# Function to undistorted the image
def cal_undistort(img, objpoints, imgpoints):
    # Use cv2.calibrateCamera() and cv2.undistort()
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
    undist = cv2.undistort(img, mtx, dist,None,mtx)
    return undist

