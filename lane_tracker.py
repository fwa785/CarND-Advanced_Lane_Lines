import numpy as np
import matplotlib.pyplot as plt
import cv2

def sliding_window(img):
    #create an output image to visualize it
    out_img = np.dstack((img, img, img))

    # take a histogram of the bottom half of the image
    histogram = np.sum(img[img.shape[0]//2:, :], axis=0)

    # Find the midpoint
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # number of sliding window
    nwindow = 9
    # height of the window
    window_height = np.int(img.shape[0]//nwindow)

    # Identify the x and y positions of all nonzero pixels
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # current position
    leftx_current = leftx_base
    rightx_current = rightx_base

    # set the window width
    margin = 100
    # minimum number of pixels found to center window
    minpix = 80

    # create list for left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # step through the window one by one
    for window in range(nwindow):
        win_y_low = img.shape[0] - (window + 1) * window_height
        win_y_high = img.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw window
        cv2.rectangle(out_img, (win_xleft_low,win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 5)
        cv2.rectangle(out_img, (win_xright_low,win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 5)

        # identify the non-zero pixels
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) &
                          (nonzerox <  win_xleft_high)).nonzero()[0]

        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xright_low) &
                          (nonzerox <  win_xright_high)).nonzero()[0]

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if(len(good_left_inds) > minpix):
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))

        if (len(good_right_inds) > minpix):
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    return out_img, leftx, rightx, lefty, righty
