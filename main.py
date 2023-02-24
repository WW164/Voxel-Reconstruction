import cv2 as cv
import os
import random
import numpy as np

CellWidth = 8
CellHeight = 6
fourCornerCoordinates = []
manual_points_entered = False
# Arrays to store object points and image points from all the images.
objPoints = []  # 3d point in real world space
imgPoints = []  # 2d points in image plane.
objP = np.zeros((CellWidth * CellHeight, 3), np.float32)
objP[:, :2] = np.mgrid[0:CellHeight, 0:CellWidth].T.reshape(-1, 2)


def interpolate_grid(coordinates, image):
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    rows, cols, ch = image.shape
    pts1 = np.float32(coordinates)
    pts2 = np.float32([[0, 0], [rows, 0], [0, cols], [rows, cols]])
    M = cv.getPerspectiveTransform(pts1, pts2)
    p2 = pts2[1]
    p3 = pts2[2]
    p4 = pts2[3]

    horizontalVector = np.subtract(p4, p2)
    verticalVector = np.subtract(p4, p3)

    grid = []
    for x in range(CellHeight):
        alpha = x / (CellHeight - 1)
        x_offset = (horizontalVector * alpha)[1:]
        for y in range(CellWidth):
            beta = y / (CellWidth - 1)
            y_offset = (verticalVector * beta)[:1]
            grid.append(tuple((y_offset, x_offset)))

    _, IM = cv.invert(M)
    reprojectedPoints = []
    for point in grid:
        x1, y1 = point
        coord = [x1, y1] + [1]
        P = np.array(coord, dtype=object)
        x, y, z = np.dot(IM, P)
        # Divide x and y by z to get 2D values
        reproj_x = x / z
        reproj_y = y / z
        reprojectedPoints.append((reproj_x, reproj_y))

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    corners2 = cv.cornerSubPix(gray, np.array(reprojectedPoints), (11, 11), (-1, -1), criteria)
    #corners2 = np.array(reprojectedPoints)
    corners2 = np.reshape(corners2, (48, 1, 2))

    objPoints.append(objP)
    imgPoints.append(corners2)
    cv.drawChessboardCorners(image, (CellHeight, CellWidth), corners2, True)
    cv.imshow('img', image)
    cv.waitKey(500)
    global manual_points_entered
    manual_points_entered = True


def click_event(event, x, y, flags, params):
    global fourCornerCoordinates

    if event == cv.EVENT_LBUTTONDOWN:

        if len(fourCornerCoordinates) < 4:
            fourCornerCoordinates.append([x, y])

        if len(fourCornerCoordinates) == 4:
            cv.destroyAllWindows()
            interpolate_grid(fourCornerCoordinates, params)
            fourCornerCoordinates = []


        img = cv.circle(params, (int(x), int(y)), 5, (255, 0, 0), 2)
        cv.imshow('img', img)


def findCorners(sampleImage):
    global manual_points_entered
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)


    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    manual_points_entered = False
    gray = cv.cvtColor(sampleImage, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, (CellHeight, CellWidth), None)
    if ret:
        objPoints.append(objP)
        accurateCorners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgPoints.append(accurateCorners)

    if not ret:
        cv.imshow('img', sampleImage)
        cv.setMouseCallback('img', click_event, param=sampleImage)
        while not manual_points_entered:
            cv.imshow('img', sampleImage)
            cv.waitKey(500)
    return objPoints, imgPoints


def findCameraIntrinsic():

    for i in range(4):
        randomFrameNumbers = set()

        camFolder = "cam" + str(i+1)
        os.chdir(os.path.join("data", camFolder))
        #os.chdir("data\cam1")
        videoName = "intrinsics.avi"
        video = cv.VideoCapture(videoName)

        totalFrames = video.get(cv.CAP_PROP_FRAME_COUNT)
        print("Total frame is: ", totalFrames)

        sample = int(totalFrames * 0.01)

        for j in range(sample):
            randomFrameNumbers.add(random.randint(0, totalFrames))
        print("Random frame number size is:", len(randomFrameNumbers))

        for randomFrame in randomFrameNumbers:
            video.set(cv.CAP_PROP_POS_FRAMES, randomFrame)
            success, image = video.read()
            if success:
                gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
                objectPoints, imagePoints = findCorners(image)
                if len(objectPoints) != 0 and len(imagePoints) != 0:
                    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objectPoints, imagePoints, gray.shape[::-1], None,                                              None)
                    print("calibrated")

        np.savez('camera_matrix', mtx=mtx, dist=dist)
        print("saved")

        os.chdir("..")
        os.chdir("..")


if __name__ == '__main__':
    findCameraIntrinsic()
