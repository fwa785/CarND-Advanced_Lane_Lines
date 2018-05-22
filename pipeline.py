import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from undistort_img import load_cal_dist, cal_undistort
from gradient import abs_sobel_thresh, mag_thresh, dir_thresh
from color_sel import color_hls_thresh
from transform_img import warp_transform
from lane_tracker import sliding_window

# Load the calibration result
objpoints, imgpoints = load_cal_dist()

# Load the test images
images = glob.glob('test_images/*.jpg')

for idx, fname in enumerate(images):
    orig_image = mpimg.imread(fname)

    # undistorted the image
    image = cal_undistort(orig_image, objpoints, imgpoints)

    # Apply each of the gradient threshold
    kernel_size = 7
    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=kernel_size, thresh=(20, 255))
    grady = abs_sobel_thresh(image, orient='y', sobel_kernel=kernel_size, thresh=(20, 255))

    # combined gradient image
    grad_binary = np.zeros_like(gradx)
    grad_binary[(gradx==1) & (grady==1)] = 1

    # select color
    color_binary = color_hls_thresh(image, s_thresh=(100, 255), h_thresh=(20, 100), l_thresh=(35,255))

    # Combine gradient and color binary
    combined = np.zeros_like(color_binary)
    combined[(color_binary == 1) | (grad_binary == 1)] = 1

    # save the gradient and color selected image
    output_filename = 'output_images/grad_color/'+fname.split('\\')[-1]
    mpimg.imsave(output_filename, combined, cmap='gray')

    # Select the region for warp transform
    transformed, M, Minv = warp_transform(combined, 0.08, 0.80, 0.62, 0.95, 0.20)

    # save the transformed image
    output_filename = 'output_images/transform/'+fname.split('\\')[-1]
    mpimg.imsave(output_filename, transformed, cmap='gray')

    out_img, left_fit, right_fit = sliding_window(transformed)

    ploty = np.linspace(0, out_img.shape[0]-1, out_img.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    warp_zero = np.zeros_like(transformed).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))

    # Combine the result with the original image
    lanemarked = cv2.addWeighted(image, 1, newwarp, 0.3, 0)

    # save the image with lane areas
    output_filename = 'output_images/lane_marked/'+fname.split('\\')[-1]
    mpimg.imsave(output_filename, lanemarked)
