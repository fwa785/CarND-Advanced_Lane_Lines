import numpy as np
import cv2

def color_hls_thresh(img, s_thresh=(0,255), h_thresh=(0,255), l_thresh=(0, 255)):
    # Convert to HLS color space and separate V channels
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    h_channel = hls[:,:,0]
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]

    # Threshold color channel
    color_binary = np.zeros_like(s_channel)
    color_binary[(s_channel > s_thresh[0]) & (s_channel < s_thresh[1]) &
                 (l_channel > l_thresh[0]) & (l_channel < l_thresh[1])] = 1

    color_binary[(h_channel > h_thresh[0]) & (h_channel < h_thresh[1]) &
                 (l_channel > l_thresh[0]) & (l_channel < l_thresh[1])] = 1

    return color_binary

def color_thresh(img, s_thresh=(0,255), v_thresh=(0,255)):
    # Convert to HLS color space and separate channels
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    h_channel = hls[:,:,0]
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]

    # Convert to HSV color space and separate channels
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    v_channel = hsv[:,:,2]

    # Threshold color channel
    color_binary = np.zeros_like(s_channel)
    color_binary[(s_channel > s_thresh[0]) & (s_channel < s_thresh[1]) &
                 (v_channel > v_thresh[0]) & (v_channel < v_thresh[1])] = 1

    return color_binary
