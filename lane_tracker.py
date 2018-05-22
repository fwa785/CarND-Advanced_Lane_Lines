import numpy as np
import matplotlib.pyplot as plt
import glob
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

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    return out_img, left_fit, right_fit

def window_mask(width, height, img_ref, center, level):
    y_low = img_ref.shape[0] - (level + 1) * height
    y_high = img_ref.shape[0] - level * height

    x_low = max(0,int(center-width/2))
    x_high = min(int(center+width/2),img_ref.shape[1])

    output = np.zeros_like(img_ref)
    output[y_low:y_high,x_low:x_high] = 1
    return output

def find_window_centroids(img, window_width, nwindow, margin):

    img_height = img.shape[0]
    img_width = img.shape[1]
    window_height = int(img_height/nwindow)
    #offset = window_width / 2
    offset = 0

    window_centroids = []
    window = np.ones(window_width)

    midpoint_x = int(img_width/2.)
    quarterpoint_y =  int(3*img_height/4.)

    #sum quarter bottom of image to get slice
    l_sum = np.sum(img[quarterpoint_y:, :midpoint_x], axis=0)
    l_center = np.argmax(np.convolve(window, l_sum)) - offset
    r_sum = np.sum(img[quarterpoint_y:, midpoint_x:], axis=0)
    r_center = np.argmax(np.convolve(window, r_sum)) - offset + midpoint_x

    # Add what we found for the first layer
    window_centroids.append((l_center, r_center))

    print(l_center, r_center)

    # Go through each layer looking for max pixel locations
    for window in range(1, nwindow):
        win_y_low = img_height - (window + 1) * window_height
        win_y_high = img_height - window * window_height

        image_layer = np.sum(img[win_y_low:win_y_high, :], axis=0)
        conv_signal = np.convolve(window, image_layer)

        #plt.plot(conv_signal)
        #plt.show()

        l_min_index = int(max(l_center + offset - margin, 0))
        l_max_index = int(min(l_center + offset + margin, img_width))
        l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset

        r_min_index = int(max(r_center + offset - margin, 0))
        r_max_index = int(min(r_center +offset + margin, img_width))
        r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset

        window_centroids.append((l_center, r_center))

    return window_centroids

def conv_sliding_window(img):

    nwindow = 5
    img_height = img.shape[0]
    img_width = img.shape[1]
    window_height = int(img_height/nwindow)
    window_width = 100
    margin = 50

    img_mask = np.zeros_like(img)
    img_mask[img == 255] = 1

    out_img = np.dstack((img_mask, img_mask, img_mask)) * 255

    window_centroids = find_window_centroids(img_mask, window_width, nwindow, margin)

    if (len(window_centroids) > 0):
        l_points = np.zeros_like(img)
        r_points = np.zeros_like(img)

        for level in range(0, len(window_centroids)):
            l_mask = window_mask(window_width, window_height, img, window_centroids[level][0], level)
            r_mask = window_mask(window_width, window_height, img, window_centroids[level][1], level)
            # Add graphic points from window mask here to total pixels found
            l_points[(l_points == 255) | ((l_mask == 1))] = 255
            r_points[(r_points == 255) | ((r_mask == 1))] = 255

        template = np.array(l_points + r_points, np.uint8)
        zero_channel = np.zeros_like(template)
        template = np.array(cv2.merge((zero_channel,template,zero_channel)), np.uint8)
        out_img = cv2.addWeighted(out_img, 1, template, 0.5, 0.0)

    return out_img
'''
# Load the test images
images = glob.glob('output_images/transform//*.jpg')

for idx, fname in enumerate(images):
    image = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)

    out_img, left_fit, right_fit = sliding_window(image)

    ploty = np.linspace(0, out_img.shape[0]-1, out_img.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    warp_zero = np.zeros_like(out_img).astype(np.uint8)
    color_warp = np.dstack(warp_zero, warp_zero, warp_zero)
    pts_left = np.array([np.transpose(np.vstack(left_fitx, ploty))])
    pts_right = np.array([np.transpose(np.vstack(right_fitx, ploty))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    plt.imshow(result)

    f = plt.figure()
    plt.imshow(out_img)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')

    #save the image
    output_filename = 'output_images/sliding_window/'+fname.split('\\')[-1]
    f.savefig(output_filename)

    #out_img = conv_sliding_window(image)
'''