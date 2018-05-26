import numpy as np
import cv2


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def warp_transform(img, upper_w, lower_w, upper_h, lower_h, offset):

    # Grab the image height and width
    img_width = img.shape[1]
    img_height = img.shape[0]
    img_size = (img_width, img_height)

    src = np.float32([[img_width * (0.5 - upper_w / 2), img_height * upper_h],
                      [img_width * (0.5 + upper_w / 2), img_height * upper_h],
                      [img_width * (0.5 + lower_w / 2), img_height * lower_h],
                      [img_width * (0.5 - lower_w / 2), img_height * lower_h]])

    dest = np.float32([[img_width * offset, 0],
                       [img_width * (1-offset), 0],
                       [img_width * (1-offset), img_height],
                       [img_width * offset, img_height]])

    M = cv2.getPerspectiveTransform(src, dest)

    Minv = cv2.getPerspectiveTransform(dest, src)

    polygon_points = [(img_width * (0.4 - upper_w / 2), img_height * upper_h),
                      (img_width * (0.6 + upper_w / 2), img_height * upper_h),
                      (img_width * (0.4 + lower_w / 2), img_height * lower_h),
                      (img_width * (0.6 - lower_w / 2), img_height * lower_h)]

    # First select the region of interest
    img = region_of_interest(img, np.array([polygon_points], np.int32))

    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

    return warped, M, Minv

