import cv2 as cv
import os
import numpy as np
import random
import argparse

Results = []
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

    for i in range(4):

        camFolder = "cam" + str(i + 1)
        os.chdir(os.path.join("data", camFolder))
        videoName = "video.avi"
        video = cv.VideoCapture(videoName)
        success, image = video.read()

        if success:
            hsvFrame = cv.cvtColor(image, cv.COLOR_BGR2HSV)
            dFrame = cv.absdiff(hsvFrame, backgroundModels[camFolder])
            dFrame = cv.cvtColor(dFrame, cv.COLOR_BGR2GRAY)

        img = RefineOutput(dFrame)
        filepath = 'foreground.png'
        print(filepath)
        if not cv.imwrite(filepath, img):
            print("failed")
        os.chdir("..")
        os.chdir("..")


def RefineOutput(image):
    global src
    src = image
    if src is None:
        print('Could not open or find the image: ', image)
        exit(0)

    cv.namedWindow(title_refine_window)
    Apply_Threshold(0)
    contours = 1000000;
    while contours != 1:
        contours = Apply_Contours(0)

    cv.createTrackbar(erode_trackbar_element_shape, title_refine_window, 0, max_elem, erosion)
    cv.createTrackbar(erode_trackbar_kernel_size, title_refine_window, 0, max_kernel_size, erosion)
    cv.createTrackbar(dilate_trackbar_element_shape, title_refine_window, 0, max_elem, dilatation)
    cv.createTrackbar(dilate_trackbar_kernel_size, title_refine_window, 0, max_kernel_size, dilatation)
    erosion(0)
    dilatation(0)
    while True:
        k = cv.waitKey(0)
        if k == ord('e'):  # e key to save
            src = temp
            print("saved")
        elif k == ord('q'):  # q key to commit the image to the list
            src = temp
            Results.append(src)
            print("committed")
            break
        else:  # normally -1 returned,so don't print it
            break
    print('returning src')
    return src


def Apply_Threshold(val):
    global src, temp
    # Threshold to binarize
    th, dFrame = cv.threshold(src, 30, 255, cv.THRESH_BINARY)
    src = dFrame
    #temp = dFrame
    cv.imshow(title_refine_window, src)


def Apply_Contours(val):
    global src, temp
    contour_value = cv.getTrackbarPos(Contour_trackbar, title_refine_window)
    contours, hierarchy = cv.findContours(src, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    img = src.copy()
    sorted_contours = sorted(contours, key=cv.contourArea, reverse=True)
    largest_item = sorted_contours[0]

    mask = np.ones(img.shape[:2], dtype="uint8") * 255
    print(len(sorted_contours))
    for i in range(len(sorted_contours)):
        if i != 0:
            cv.drawContours(mask, sorted_contours[i], -1, 0, -1)
    # remove the contours from the image and show the resulting images
    image = cv.bitwise_and(img, img, mask=mask)
    cv.imshow(title_refine_window, image)
    src = image
    return len(sorted_contours)


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
