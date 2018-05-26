import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from undistort_img import load_cal_dist, cal_undistort
from gradient import abs_sobel_thresh, mag_thresh, dir_thresh
from color_sel import color_hls_thresh, color_thresh
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

    # combined gradient image with AND
    grad_binary = np.zeros_like(gradx)
    grad_binary[(gradx==1) & (grady==1)] = 1

    # select color
    color_binary = color_hls_thresh(image, s_thresh=(100, 255), h_thresh=(20, 100))

    # Combine gradient and color binary with OR
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

    lane_found_img, leftx, rightx, lefty, righty = sliding_window(transformed)

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    ploty = np.linspace(0, lane_found_img.shape[0]-1, lane_found_img.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # draw the fitting curves
    f = plt.figure()
    plt.imshow(lane_found_img)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')

    #save the image
    output_filename = 'output_images/sliding_window/'+fname.split('\\')[-1]
    f.savefig(output_filename)

    # Mark the lane area
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

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 40 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    # fit in the world space
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)

    # the car's position in world space
    car_y = lane_found_img.shape[0]*ym_per_pix
    car_x = (lane_found_img.shape[1]*xm_per_pix)/2.

    # calculate the radius of curvature
    left_curverad = ((1+((2*left_fit_cr[0]*car_y+left_fit_cr[1])**2))**1.5)/np.absolute(2*left_fit_cr[0])
    right_curverad = ((1+((2*right_fit_cr[0]*car_y+right_fit_cr[1])**2))**1.5)/np.absolute(2*right_fit_cr[0])
    cv2.putText(lanemarked, 'Radius of Curvature =' + str(round(left_curverad, 3)) +'(m)', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))

    # calculate the left x and right x using the world space fitting
    left_fitx_lane = left_fit_cr[0]*(car_y**2) + left_fit_cr[1]*car_y + left_fit_cr[2]
    right_fitx_lane = right_fit_cr[0]*(car_y**2) + right_fit_cr[1]*car_y + right_fit_cr[2]

    # calculate the position of the center of the lanes
    lane_center = (left_fitx_lane + right_fitx_lane)/2.

    # calculate the offset of the center of the image
    car_offset = (lane_center - car_x)
    cv2.putText(lanemarked, 'Position from the center to the left ' + str(round(car_offset, 3)) +'(m)', (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))

    # save the image with lane areas
    output_filename = 'output_images/lane_marked/'+fname.split('\\')[-1]
    mpimg.imsave(output_filename, lanemarked)
