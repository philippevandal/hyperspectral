# !/usr/bin/python
import sys, traceback
import cv2
import numpy as np
from plantcv import plantcv as pcv

import time
import picamera

# OpenCV setup
cv2.namedWindow("plantcv", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("plantcv", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# screen = np.zeros((480,640,3), np.uint8)

font                   = cv2.FONT_HERSHEY_SIMPLEX
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 2

while True:
    # Picamera directly to openCV numpy.array
    with picamera.PiCamera() as camera:
        camera.resolution = (320, 240)
        camera.framerate = 24
        time.sleep(2)
        image = np.empty((240, 320, 3), dtype=np.uint8)
        camera.capture(image, 'rgb')

    # Create masked image from a color image based LAB color-space and threshold values.
    # for lower and upper_thresh list as: thresh = [L_thresh, A_thresh, B_thresh]
    mask, masked_img = pcv.threshold.custom_range(rgb_img=image, lower_thresh=[0,0,158], upper_thresh=[255,255,255], channel='LAB')

    # Read image (readimage mode defaults to native but if image is RGBA then specify mode='rgb')
    # Inputs:
    #   filename - Image file to be read in 
    #   mode     - Return mode of image; either 'native' (default), 'rgb', 'gray', 'envi', or 'csv'
    spectral_array = pcv.readimage(filename=image, mode='envi')

    filename = spectral_array.filename

    # Save the pseudo-rgb image that gets created while reading in hyperspectral data
    pcv.print_image(img=spectral_array.pseudo_rgb, filename=filename + "_pseudo-rgb.png")

    # Extract the Green Difference Vegetation Index from the datacube 
    index_array_gdvi  = pcv.spectral_index.gdvi(hsi=spectral_array, distance=20)

    # Threshold the grayscale image 
    gdvi_thresh = pcv.threshold.binary(gray_img=index_array_gdvi.array_data, threshold=150, max_value=255)

    # Define ROI 
    roi, roi_hierarchy= pcv.roi.rectangle(img=gdvi_thresh, x=0, y=0, h=500, w=500)

    # Find Objects 
    id_objects, obj_hierarchy = pcv.find_objects(img=index_array_gdvi.array_data, mask=gdvi_thresh)

    # Filter object by a defined region of interest 
    roi_objects, hierarchy, kept_mask, obj_area = pcv.roi_objects(img=index_array_gdvi.array_data, roi_contour=roi, 
                                                                  roi_hierarchy=roi_hierarchy, 
                                                                  object_contour=id_objects, 
                                                                  obj_hierarchy=obj_hierarchy,
                                                                  roi_type='partial')

    # Apply the mask of the leaf to the entire datacube, and store it where the datacube is stored.
    spectral_array.array_data = pcv.apply_mask(rgb_img=spectral_array.array_data, mask=kept_mask, mask_color="black")
    print(spectral_array.array_data)
    # Extract reflectance intensity data and store it out to the Outputs class. 
    analysis_img = pcv.hyperspectral.analyze_spectral(array=spectral_array, mask=kept_mask, histplot=True)

    # Extract statistics about an index for the leaf region 
    pcv.hyperspectral.analyze_index(array=index_array_gdvi, mask=kept_mask)

    # Write shape and color data to results file
    #pcv.print_results(filename=args.result)

    # A picture is taken and analyzed every minute
    time.sleep(60)

    # GPIOs cleanup and exiting the sketch with "q" (or possibly a button?)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        GPIO.cleanup()
        cv2.destroyAllWindows()
        break