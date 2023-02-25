import cv2 as cv
import os
import numpy as np
import random


def createBackgroundModel():
    frames = []
    medianFrames = {}

    for i in range(4):

        randomFrameNumbers = set()

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

        medianFrame = np.median(frames, axis=0)

        medianFrames[camFolder] = medianFrame

        os.chdir("..")
        os.chdir("..")

    return medianFrames


def backgroundSubtraction(backgroundModels):
    backSub = cv.createBackgroundSubtractorKNN()

    # for medianFrame in backgroundModels:
    #     grayMedianFrame.append(cv.cvtColor(medianFrame, cv.COLOR_BGR2GRAY))

    for i in range(4):

        camFolder = "cam" + str(i + 1)
        os.chdir(os.path.join("data", camFolder))
        videoName = "video.avi"
        video = cv.VideoCapture(videoName)
        success, image = video.read()

        while success:
            hsvFrame = cv.cvtColor(image, cv.COLOR_BGR2HSV)
            cv.imshow("hsv", hsvFrame)
            cv.waitKey(0)
            dFrame = cv.absdiff(hsvFrame, backgroundModels[camFolder])
            print(dFrame)

        os.chdir("..")
        os.chdir("..")
