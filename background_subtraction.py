import cv2 as cv
import os
import numpy as np
import random
import argparse

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

def createBackgroundModel():
    frames = []
    medianFrames = {}

    for i in range(4):

        randomFrameNumbers = set()
        frames = []
        camFolder = "cam" + str(i + 1)
        os.chdir(os.path.join("data", camFolder))
        videoName = "background.avi"
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
        medianFrames[camFolder] = medianFrame

        os.chdir("..")
        os.chdir("..")

    return medianFrames


def backgroundSubtraction(backgroundModels):
    backSub = cv.createBackgroundSubtractorKNN()

    # for medianFrame in backgroundModels:
    #     grayMedianFrame.append(cv.cvtColor(medianFrame, cv.COLOR_BGR2GRAY))
    image = []
    for i in range(4):

        camFolder = "cam" + str(i + 1)
        os.chdir(os.path.join("data", camFolder))
        videoName = "video.avi"
        video = cv.VideoCapture(videoName)
        success, image = video.read()

        if success:
            #hsvFrame = np.float32(cv.cvtColor(image, cv.COLOR_BGR2HSV))
            hsvFrame = cv.cvtColor(image, cv.COLOR_BGR2HSV)
            #print(type(hsvFrame[0][0][0]))

            #print(len(backgroundModels))

            dFrame = cv.absdiff(hsvFrame, backgroundModels[camFolder])
            dFrame = cv.cvtColor(dFrame, cv.COLOR_BGR2GRAY)
            # # Threshold to binarize
            # th, dFrame = cv.threshold(dFrame, 50, 255, cv.THRESH_BINARY)
            image = dFrame
            #cv.imshow("hsv",  dFrame)
            #cv.waitKey(0)
            #print(dFrame)
        os.chdir("..")
        os.chdir("..")
    RefineOutput(image)


def RefineOutput(image):
    global src
    src = image
    if src is None:
        print('Could not open or find the image: ', image)
        exit(0)

    cv.namedWindow(title_refine_window)

    cv.createTrackbar(Threshold_trackbar, title_refine_window, 10, 100, Apply_Threshold)
    cv.createTrackbar(Contour_trackbar, title_refine_window, 0, 91, Apply_Contours)
    cv.createTrackbar(erode_trackbar_element_shape, title_refine_window, 0, max_elem, erosion)
    cv.createTrackbar(erode_trackbar_kernel_size, title_refine_window, 0, max_kernel_size, erosion)
    cv.createTrackbar(dilate_trackbar_element_shape, title_refine_window, 0, max_elem, dilatation)
    cv.createTrackbar(dilate_trackbar_kernel_size, title_refine_window, 0, max_kernel_size, dilatation)
    erosion(0)
    dilatation(0)
    #cv.waitKey()
    while True:
        k = cv.waitKey(0)
        if k == ord('e'):  # Esc key to stop
            src = temp
            print("saved")
        else:  # normally -1 returned,so don't print it
            break


def Apply_Threshold(val):
    global src, temp
    threshold_value = cv.getTrackbarPos(Threshold_trackbar, title_refine_window)
    # Threshold to binarize
    th, dFrame = cv.threshold(src, threshold_value, 255, cv.THRESH_BINARY)
    temp = dFrame
    cv.imshow(title_refine_window, dFrame)


def Apply_Contours(val):
    global src, temp
    contour_value = cv.getTrackbarPos(Contour_trackbar, title_refine_window)
    contours, hierarchy = cv.findContours(src, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    print(len(contours))
    #contour = contours[0] if len(contours) == 2 else contours[1]
    #big_contour = max(contours, key=cv.contourArea)

    img = src.copy()
    cv.drawContours(image=img, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv.FILLED)
    cv.imshow(title_refine_window, img)
    temp = img


def morph_shape(val):
    if val == 0:
        return cv.MORPH_RECT
    elif val == 1:
        return cv.MORPH_CROSS
    elif val == 2:
        return cv.MORPH_ELLIPSE


def erosion(val):
    global src, temp
    erosion_size = cv.getTrackbarPos(erode_trackbar_kernel_size, title_refine_window)
    erosion_shape = morph_shape(cv.getTrackbarPos(erode_trackbar_element_shape, title_refine_window))

    element = cv.getStructuringElement(erosion_shape, (2 * erosion_size + 1, 2 * erosion_size + 1),
                                       (erosion_size, erosion_size))
    erode_dst = cv.erode(src, element)
    cv.imshow(title_refine_window, erode_dst)
    temp = erode_dst


def dilatation(val):
    global src, temp
    dilatation_size = cv.getTrackbarPos(dilate_trackbar_kernel_size, title_refine_window)
    dilation_shape = morph_shape(cv.getTrackbarPos(dilate_trackbar_element_shape, title_refine_window))
    element = cv.getStructuringElement(dilation_shape, (2 * dilatation_size + 1, 2 * dilatation_size + 1),
                                       (dilatation_size, dilatation_size))
    dilate_dst = cv.dilate(src, element)
    cv.imshow(title_refine_window, dilate_dst)
    temp = dilate_dst


def backgroundSubtractionKNN():
    backSub = cv.createBackgroundSubtractorKNN()
    for i in range(4):

        randomFrameNumbers = set()
        frames = []
        camFolder = "cam" + str(i + 1)
        os.chdir(os.path.join("data", camFolder))
        videoName = "background.avi"
        video = cv.VideoCapture(videoName)
        if not video.isOpened():
            print('Unable to open: ' + videoName)
            exit(0)
        while True:
            ret, frame = video.read()
            if frame is None:
                break

            fgMask = backSub.apply(frame)

            cv.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
            cv.putText(frame, str(video.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

            cv.imshow('Frame', frame)
            cv.imshow('FG Mask', fgMask)

            keyboard = cv.waitKey(30)
            if keyboard == 'q' or keyboard == 27:
                break

