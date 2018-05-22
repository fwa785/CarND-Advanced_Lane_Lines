import numpy as np
import cv2

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0,100)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Take the derivative in x or y given orient = 'x' or 'y'
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # Take the absolute value of derivative or gradient
    abs_sobel = np.absolute(sobel)

    # scale to 8-bit(0-255) then convert to unit8
    scaled_sobel = np.uint8(255.*abs_sobel/np.max(abs_sobel))

    # creat a mask of 1's where the scaled gradient maginitude
    # is > thresh_min and < thresh_max
    sobel_mask = np.zeros_like(scaled_sobel)
    sobel_mask[(scaled_sobel > thresh[0]) & (scaled_sobel < thresh[1])] = 1

    return sobel_mask

# Define a function that applies Sobel x and y
# then computes the magnitude of the gradient
# and applies a threshold
def mag_thresh(img, sobel_kernel=3, thresh=(0,255)):
    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Take the gradient of x and y separately
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # calculate the magitude
    sobel = np.sqrt(np.square(sobel_x) + np.square(sobel_y))

    # Scale to 8 bits
    scaled_sobel = np.uint8(255*sobel/np.max(sobel))

    # Create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel > thresh[0]) & (scaled_sobel < thresh[1])] = 1

    return binary_output

# Define a function that applies Sobel x and y
# the compute the direction of the gradient
# and applies a threshold
def dir_thresh(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # convery to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Take the gradient in x and y separately
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # Take the absolute value of the x and y gradient
    abs_sobel_x = np.absolute(sobel_x)
    abs_sobel_y = np.absolute(sobel_y)

    # Calculate the direction of the gradient
    dir_sobel = np.arctan2(abs_sobel_y, abs_sobel_x)

    # Create binary mask where direction thresholds are met
    binary_output = np.zeros_like(dir_sobel)
    binary_output[(dir_sobel > thresh[0]) & (dir_sobel < thresh[1])] = 1

    return binary_output
