import cv2 as cv
import os
import random
import numpy as np

CellWidth = 8
CellHeight = 6

#ToDo: Refactor in a way that this returns corners
def interpolate_grid(coordinates, image):
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
    for x in range(CellWidth):
        alpha = x / (CellWidth - 1)
        x_offset = (horizontalVector * alpha)[1:]
        for y in range(CellHeight):
            beta = y / (CellHeight - 1)
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
    objPoints.append(objP)

    global corners2
    corners2 = cv.cornerSubPix(gray, np.array(reprojectedPoints), (11, 11), (-1, -1), criteria)
    corners2 = np.reshape(corners2, (54, 1, 2))
    imgPoints.append(corners2)
    cv.drawChessboardCorners(image, (CellHeight, CellWidth), corners2, True)
    global manual_points_entered
    manual_points_entered = True


def click_event(event, x, y, flags, params):
    fourCornerCoordinates = []

    if event == cv.EVENT_LBUTTONDOWN:

        if len(fourCornerCoordinates) < 4:
            fourCornerCoordinates.append([x, y])

        if len(fourCornerCoordinates) == 4:
            return fourCornerCoordinates
            pass

        img = cv.circle(params, (int(x), int(y)), 5, (255, 0, 0), 2)
        cv.imshow('img', img)


def findCorners(sampleImage):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objP = np.zeros((CellWidth * CellHeight, 3), np.float32)
    objP[:, :2] = np.mgrid[0:CellHeight, 0:CellWidth].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objPoints = []  # 3d point in real world space
    imgPoints = []  # 2d points in image plane.

    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    manual_points_entered = False
    gray = cv.cvtColor(sampleImage, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, (CellHeight, CellWidth), None)
    if ret:
        objPoints.append(objP)
        accurateCorners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgPoints.append(accurateCorners)

    #ToDo: if you uncomment this statement program won't work for now

    # if not ret:
    #     cv.imshow('img', sampleImage)
    #     cv.setMouseCallback('img', click_event, param=sampleImage)
    #     while not manual_points_entered:
    #         cv.imshow('img', sampleImage)
    #         cv.waitKey(0)
    #
    #     #ToDo: This captures the event return value
    #     fourCornerCoordinates = []
    #     while len(fourCornerCoordinates) < 4:
    #         key = cv.waitKey(1)
    #         if key == ord('q'):
    #             break
    #         elif key == ord('c'):
    #             coords = click_event(None, None, None, None, sampleImage)
    #             if coords is not None:
    #                 fourCornerCoordinates = coords
    #     #ToDo: This uses the corners found manually and the image to find other corners and returns ... (what do you think it
    #      should return to match the other part of the code?)
    #     interpolate_grid(fourCornerCoordinates, sampleImage)

    return objPoints, imgPoints


def findCameraIntrinsic():
    randomFrameNumbers = set()

    os.chdir("data\cam1")
    videoName = "intrinsics.avi"
    video = cv.VideoCapture(videoName)

    totalFrames = video.get(cv.CAP_PROP_FRAME_COUNT)
    print("Total frame is: ", totalFrames)

    sample = int(totalFrames * 0.01)

    for i in range(sample):
        randomFrameNumbers.add(random.randint(0, totalFrames))
    print("Random frame number size is:", len(randomFrameNumbers))

    for randomFrame in randomFrameNumbers:
        video.set(cv.CAP_PROP_POS_FRAMES, randomFrame)
        success, image = video.read()
        if success:
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            objectPoints, imagePoints = findCorners(image)
            if len(objectPoints) != 0 and len(imagePoints) != 0:
                ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objectPoints, imagePoints, gray.shape[::-1], None,
                                                                  None)
                print("calibrated")


if __name__ == '__main__':
    findCameraIntrinsic()
