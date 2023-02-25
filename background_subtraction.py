import cv2 as cv
import os
import numpy as np
import random

def createBackgroundModel():
    frames = []
    medianFrames = {}
    backSub = cv.createBackgroundSubtractorKNN()

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
                frames.append(image)

        medianFrame = np.median(frames, axis=0)

        medianFrames[camFolder] = medianFrame

        os.chdir("..")
        os.chdir("..")

    return medianFrames


def backgroundSubtraction(backgroundModels):
    grayMedianFrame = []

    for medianFrame in backgroundModels:
        grayMedianFrame.append(cv.cvtColor(medianFrame, cv.COLOR_BGR2GRAY))



