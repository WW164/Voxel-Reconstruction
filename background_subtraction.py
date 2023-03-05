import cv2 as cv
import os
import numpy as np
import random
import argparse

Results = []
backgroundModels = []
src = None
temp = None
erosion_size = 0
max_elem = 2
max_kernel_size = 21
Threshold_trackbar = "Threshold Value"
Contour_trackbar = "Contour Value"
erode_trackbar_element_shape = 'Erode Element:\n 0: Rect \n 1: Cross \n 2: Ellipse'
erode_trackbar_kernel_size = 'Erode Kernel size:\n 2n +1'
dilate_trackbar_element_shape = 'Dilate Element:\n 0: Rect \n 1: Cross \n 2: Ellipse'
dilate_trackbar_kernel_size = 'Dilate Kernel size:\n 2n +1'
title_refine_window = 'Refine Values'


# Create the background model using the background.avi video
def createBackgroundModel():
    medianFrames = {}

    for i in range(4):

        randomFrameNumbers = set()
        frames = []
        camFolder = "cam" + str(i + 1)
        filepath = os.path.join("data", camFolder)
        videoName = os.path.join(filepath, "background.avi")
        video = cv.VideoCapture(videoName)

        totalFrames = video.get(cv.CAP_PROP_FRAME_COUNT)

        sample = int(totalFrames * 0.2)

        for j in range(sample):
            randomFrameNumbers.add(random.randint(0, totalFrames))

        for randomFrame in randomFrameNumbers:
            video.set(cv.CAP_PROP_POS_FRAMES, randomFrame)
            success, image = video.read()
            if success:
                hsvImage = cv.cvtColor(image, cv.COLOR_BGR2HSV)
                frames.append(hsvImage)

        medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)
        medianFrames[i] = medianFrame

    global backgroundModels
    backgroundModels = medianFrames


# Convert the image to get it ready for background subtraction.
def backgroundSubtraction(frame, cameraIndex):
    global backgroundModels
    hsvFrame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    dFrame = cv.absdiff(hsvFrame, backgroundModels[cameraIndex])
    dFrame = cv.cvtColor(dFrame, cv.COLOR_BGR2GRAY)
    dFrame = Apply_Threshold(30, dFrame)
    dFrame = Apply_Contours(dFrame)
    dFrame = dilatation(3, dFrame)
    return dFrame


# First apply a flat threshold to the image to binarize
def Apply_Threshold(val, frame):
    # Threshold to binarize
    th, dFrame = cv.threshold(frame, val, 255, cv.THRESH_BINARY)
    return dFrame


# Then recursively remove all unwanted contours until there is only 1 left.
def Apply_Contours(frame):
    contours = 1000000
    image = frame
    while contours != 1:
        contours, hierarchy = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        sorted_contours = sorted(contours, key=cv.contourArea, reverse=True)
        img = image.copy()
        mask = np.ones(img.shape[:2], dtype="uint8") * 255
        for i in range(len(sorted_contours)):
            if i != 0:
                cv.drawContours(mask, sorted_contours[i], -1, 0, -1)
        # remove the contours from the image
        image = cv.bitwise_and(img, img, mask=mask)
        contours = len(sorted_contours)

    return image


# Finally dilate the image using a flat value of 3 to fill in holes
def dilatation(val, frame):
    dilatation_size = val
    element = cv.getStructuringElement(0, (2 * dilatation_size + 1, 2 * dilatation_size + 1),
                                       (dilatation_size, dilatation_size))
    dilate_dst = cv.dilate(frame, element)
    return dilate_dst
