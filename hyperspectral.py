# !/usr/bin/python
import sys, traceback
import cv2
import numpy as np
from plantcv import plantcv as pcv
from fractions import Fraction
import time
import picamera

# OpenCV setup
cv2.namedWindow("plantcv", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("plantcv", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

font                   = cv2.FONT_HERSHEY_SIMPLEX
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 2


def readImagesAndTimes():  
    times = np.array([ 1.25, 0.2, 0.05, 0.0125 ], dtype=np.float32)

    filenames = ["ldr_01.jpg", "ldr_02.jpg", "ldr_03.jpg", "ldr_04.jpg"]

    images = []
    for filename in filenames:
        im = cv2.imread(filename)
        images.append(im)

    return images, times

while True:
    with picamera.PiCamera() as camera:
        camera.resolution = (320, 240)
        time.sleep(2)

        camera.framerate = Fraction(1, 2)
        camera.iso = 100
        camera.exposure_mode = 'off'
        camera.awb_mode = 'off'
        camera.awb_gains = (1.8,1.8)
        #0.8s exposure
        camera.framerate = 1
        camera.shutter_speed = 800000
        camera.capture('ldr_01.jpg')
        #0.2s exposure 1/5
        camera.framerate = 5
        camera.shutter_speed = 200000
        camera.capture('ldr_02.jpg')
        #0.05s exposure 1/20
        camera.framerate = 20
        camera.shutter_speed = 50000
        camera.capture('ldr_03.jpg')
        #0.0125s exposure 1/8
        camera.framerate = 30
        camera.shutter_speed = 12500
        camera.capture('ldr_04.jpg')
        #0.003125s exposure 1/320
        camera.shutter_speed = 3125
        camera.capture('ldr_05.jpg')
        #0.0008s exposure 1/12500
        camera.shutter_speed = 800
        camera.capture('ldr_06.jpg')

    images, times = readImagesAndTimes()
    # Align input images
    print("Aligning images ... ")
    alignMTB = cv2.createAlignMTB()
    alignMTB.process(images, images)

    # Obtain Camera Response Function (CRF)
    print("Calculating Camera Response Function (CRF) ... ")
    calibrateDebevec = cv2.createCalibrateDebevec()
    responseDebevec = calibrateDebevec.process(images, times)

    # Merge images into an HDR linear image
    print("Merging images into one HDR image ... ")
    mergeDebevec = cv2.createMergeDebevec()
    hdrDebevec = mergeDebevec.process(images, times, responseDebevec)
    # Save HDR image.
    cv2.imwrite("hdrDebevec.hdr", hdrDebevec)
    print("saved hdrDebevec.hdr ")

    # Read image (readimage mode defaults to native but if image is RGBA then specify mode='rgb')
    # Inputs:
    #   filename - Image file to be read in 
    #   mode     - Return mode of image; either 'native' (default), 'rgb', 'gray', 'envi', or 'csv'
    spectral_array = pcv.readimage(filename="/hdrDebevec.hdr", mode='envi')

    filename = spectral_array.filename

    # Save the pseudo-rgb image that gets created while reading in hyperspectral data
    # pcv.print_image(img=spectral_array.pseudo_rgb, filename=filename + "_pseudo-rgb.png")

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
    cv2.imshow('plantcv',analysis_img)

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
