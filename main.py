import cv2 as cv
import os
import random
import numpy as np
import background_subtraction as bs

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
    
    global corners
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    corners = cv.cornerSubPix(gray, np.array(reprojectedPoints), (11, 11), (-1, -1), criteria)
    corners = np.reshape(corners, (48, 1, 2))

    objPoints.append(objP)
    imgPoints.append(corners)
    cv.drawChessboardCorners(image, (CellWidth, CellHeight), corners, True)
    print("Draw")
    global manual_points_entered
    manual_points_entered = True


def click_event(event, x, y, flag, params):
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

    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    manual_points_entered = False
    gray = cv.cvtColor(sampleImage, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, (CellWidth, CellHeight), None)
    if ret:
        objPoints.append(objP)
        accurateCorners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgPoints.append(accurateCorners)
        cv.drawChessboardCorners(sampleImage, (CellWidth, CellHeight), accurateCorners, ret)
        cv.imshow('img', sampleImage)
        cv.waitKey(0)

    '''
    if not ret:
        cv.imshow('img', sampleImage)
        cv.setMouseCallback('img', click_event, param=sampleImage)
        while not manual_points_entered:
            cv.imshow('img', sampleImage)
            cv.waitKey(500)
    '''
    if len(objPoints) != 0 and len(imgPoints) != 0:
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objPoints, imgPoints, gray.shape[::-1], None,
                                                          None)
        ret, rvecs, tvecs = cv.solvePnP(np.float32(objPoints), accurateCorners, mtx, dist)
        axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)
        imgPts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)
        draw(sampleImage, corners, imgPts)

    return objPoints, imgPoints


def findCameraIntrinsic():
    for i in range(4):
        randomFrameNumbers = set()

        camFolder = "cam" + str(i + 1)
        os.chdir(os.path.join("data", camFolder))
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
                    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objectPoints, imagePoints, gray.shape[::-1], None,
                                                                      None)
                    print("calibrated")

        #np.savez('camera_matrix', mtx=mtx, dist=dist)
        #print("saved")

        os.chdir("..")
        os.chdir("..")


def draw(image, corners, imgPts):

    corner = tuple(corners[0].ravel().astype(int))
    img = cv.line(image, corner, tuple(imgPts[0].ravel().astype(int)), (255, 0, 0), 5)
    img = cv.line(image, corner, tuple(imgPts[1].ravel().astype(int)), (0, 255, 0), 5)
    img = cv.line(image, corner, tuple(imgPts[2].ravel().astype(int)), (0, 0, 255), 5)
    cv.imshow("img", img)
    cv.waitKey(0)


def findCameraExtrinsic():

    axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)

    for i in range(4):

        camFolder = "cam" + str(i + 1)
        os.chdir(os.path.join("data", camFolder))
        videoName = "checkerboard.avi"
        video = cv.VideoCapture(videoName)
        success, image = video.read()
        if success:
            objectPoints, imagePoints = findCorners(image)
            if len(objectPoints) != 0 and len(imagePoints) != 0:
                with np.load('camera_matrix.npz') as file:
                    intrinsicMatrix, dist = [file[i] for i in ['mtx', 'dist']]
                ret, rotation, translation = cv.solvePnP(np.float32(objectPoints), corners2, intrinsicMatrix, dist)
                print("calibrated")

        np.savez('camera_matrix_extrinsic', rvec=rotation, tvec=translation)
        print("saved")

        imgPts, jac = cv.projectPoints(axis, rotation, translation, intrinsicMatrix, dist)
        draw(image, corners2, imgPts)

        os.chdir("..")
        os.chdir("..")


if __name__ == '__main__':
    findCameraIntrinsic()
    #findCameraExtrinsic()
    backgroundModels = bs.createBackgroundModel()
    bs.backgroundSubtraction(backgroundModels)
