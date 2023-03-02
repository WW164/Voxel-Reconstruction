import cv2 as cv
import os
import random
import numpy as np
import background_subtraction as bs
import json
import csv
import functools
import operator

CellWidth = 8
CellHeight = 6
tileSize = 115

frameCellWidth = 20
frameCellHeight = 20
frameCellDepth = 20

fourCornerCoordinates = []
manual_points_entered = False
# Arrays to store object points and image points from all the images.
objPoints = []  # 3d point in real world space
imgPoints = []  # 2d points in image plane.
objP = np.zeros((CellWidth * CellHeight, 3), np.float32)
objP[:, :2] = np.mgrid[0:CellWidth, 0:CellHeight].T.reshape(-1, 2) * tileSize

framePoint = np.zeros((frameCellWidth * frameCellHeight * frameCellDepth, 3), np.float32)
framePoint[:, :3] = np.mgrid[0:frameCellWidth, 0:frameCellHeight, 0:frameCellDepth].T.reshape(-1, 3) * tileSize


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
    
    global corners2
    corners2 = np.reshape(reprojectedPoints, (48, 1, 2))

    objPoints.append(objP)
    imgPoints.append(corners2)
    #cv.drawChessboardCorners(image, (CellWidth, CellHeight), corners2, True)
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
    global corners2

    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    manual_points_entered = False
    gray = cv.cvtColor(sampleImage, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, (CellWidth, CellHeight), None)
    if ret:
        objPoints.append(objP)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgPoints.append(corners2)
        #cv.drawChessboardCorners(sampleImage, (CellWidth, CellHeight), corners2, ret)
        #cv.imshow('img', sampleImage)
        #cv.waitKey(50)


    if not ret:
        cv.imshow('img', sampleImage)
        cv.setMouseCallback('img', click_event, param=sampleImage)
        while not manual_points_entered:
            cv.imshow('img', sampleImage)
            cv.waitKey(500)


    return objPoints, imgPoints, ret


def findCameraIntrinsic():

    global corners2

    for i in range(4):
        randomFrameNumbers = set()

    camFolder = "cam4"
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
            objectPoints, imagePoints, ret = findCorners(image)
            if ret:
                ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objectPoints, imagePoints, gray.shape[::-1], None,
                                                                  None)
                print("calibrated")

    np.savez('camera_matrix', mtx=mtx, dist=dist)
    print("saved")

    with np.load('camera_matrix.npz') as file:
        intrinsicMatrix, dist = [file[i] for i in ['mtx', 'dist']]

    axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3) * tileSize
    video.set(cv.CAP_PROP_POS_FRAMES, 0)
    success, image = video.read()
    if success:
        objectPoints, imagePoints, ret = findCorners(image)
        if ret:
            ret, rotation, translation = cv.solvePnP(objP, corners2, intrinsicMatrix, dist)
            imgPts, jac = cv.projectPoints(axis, rotation, translation, intrinsicMatrix, dist)
            draw(image, corners2, imgPts)

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

    axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3) * tileSize

    for i in range(4):

        camFolder = "cam" + str(i + 1)
        os.chdir(os.path.join("data", camFolder))
        videoName = "checkerboard.avi"
        video = cv.VideoCapture(videoName)
        success, image = video.read()
        if success:
            objectPoints, imagePoints, ret = findCorners(image)
            with np.load('camera_matrix.npz') as file:
                intrinsicMatrix, dist = [file[i] for i in ['mtx', 'dist']]
            ret, rotation, translation = cv.solvePnP(objP, corners2, intrinsicMatrix, dist)
            print("calibrated")

        np.savez('camera_matrix_extrinsic', rvec=rotation, tvec=translation)
        print("saved")

        imgPts, jac = cv.projectPoints(axis, rotation, translation, intrinsicMatrix, dist)
        draw(image, corners2, imgPts)

        '''

        voxelCoordinates, jac = cv.projectPoints(framePoint, rotation, translation, intrinsicMatrix, dist)

        for point in voxelCoordinates:
            img = cv.circle(image, (int(point[0][0] - 200), int(point[0][1]) - 200), 5, (255, 0, 0), 2)

        cv.imshow('img', img)
        cv.waitKey(0)
        '''

        os.chdir("..")
        os.chdir("..")

    cv.destroyAllWindows()


def createLookupTable():
    cameraLookupTable = {}

    for i in range(4):

        camFolder = "cam" + str(i + 1)
        os.chdir(os.path.join("data", camFolder))

        with np.load('camera_matrix.npz') as file:
            intrinsicMatrix, dist = [file[i] for i in ['mtx', 'dist']]

        with np.load('camera_matrix_extrinsic.npz') as file:
            rotation, translation = [file[i] for i in ['rvec', 'tvec']]

        pointsLookupTable = {}

        for point in framePoint:
            print("point is: ", point)
            projectedPoint = cv.projectPoints(point, rotation, translation, intrinsicMatrix, dist)
            print("projected point is: ", projectedPoint[0].ravel())

            Xim = projectedPoint[0].ravel()[0]
            Yim = projectedPoint[0].ravel()[1]

            Xc = point[0]
            Yc = point[1]
            Zc = point[2]

            pointsLookupTable[str((Xc, Yc, Zc))] = str((Xim, Yim))

        cameraLookupTable[camFolder] = pointsLookupTable

        os.chdir("..")
        os.chdir("..")

    print("camera look up table is: ", cameraLookupTable)

    jsonLookupTable = json.dumps(cameraLookupTable)

    # open file for writing, "w"
    f = open("dict.json", "w")

    # write json object to file
    f.write(jsonLookupTable)

    # close file
    f.close()


def GetForeground(camera):
    camFolder = "cam" + str(camera)
    path = os.path.join("data", camFolder, 'foreground.png')
    img = cv.imread(path)
    dimensions = img.shape
    for x in range(0, dimensions[0]):
        for y in range(0, dimensions[1]):
            if np.linalg.norm(img[x, y]) > 1:
                print(x, y)


def checkVoxels():
    cameraLookupTable = {}

    #Define the range of the cube
    Xl = 0
    Xh = 7
    Yl = -4
    Yh = 6
    Zl = 2
    Zh = -13

    #save the output in result
    result = []
    #GetForeground(1)

    for i in range(4):
        #Read in the foreground image.
        voxelCoordinates = []
        camFolder = "cam" + str(i + 1)
        path = os.path.join("data", camFolder)
        foregroundImagePath = os.path.join("data", camFolder, 'foreground.png')
        foregroundImage = cv.imread(foregroundImagePath)

        #read in the camera matrix
        with np.load(os.path.join(path, 'camera_matrix.npz')) as file:
            intrinsicMatrix, dist = [file[i] for i in ['mtx', 'dist']]
        with np.load(os.path.join(path, 'camera_matrix_extrinsic.npz')) as file:
            rotation, translation = [file[i] for i in ['rvec', 'tvec']]

        for x in range(Xl, Xh):
            for y in range(Yl, Yh):
                for z in range(Zh, Zl):
                    output = []
                    # Get the projected point of the voxel position.
                    voxelPoint = np.float32((x, y, z)) * tileSize
                    voxelCoordinate, jac = cv.projectPoints(voxelPoint, rotation, translation, intrinsicMatrix, dist)
                    fx = int(voxelCoordinate[0][0][0])
                    fy = int(voxelCoordinate[0][0][1])

                    Xc = voxelPoint[0]
                    Yc = voxelPoint[1]
                    Zc = voxelPoint[2]

                    output.append((str((fy, fx))))
                    voxelCoordinates.append(voxelCoordinate)

                    if str((Xc, Yc, Zc)) in cameraLookupTable:
                        cameraLookupTable[str((Xc, Yc, Zc))].append((str((fy, fx))))
                    else:
                        cameraLookupTable[str((Xc, Yc, Zc))] = output


                    # Check if the value of the image pixel is > 1.
                    #if np.linalg.norm(foregroundImage[fy, fx]) > 1:


        # Draw the voxels for confirmation.
        for voxel in voxelCoordinates:
            img = cv.circle(foregroundImage, (int(voxel[0][0][0]), int(voxel[0][0][1])), 5, (255, 0, 0), 2)
            cv.imshow('img', img)
            cv.waitKey(1)
        result.append(voxelCoordinates)

    jsonLookupTable = json.dumps(cameraLookupTable)

    # open file for writing, "w"
    f = open("dict.json", "w")

    # write json object to file
    f.write(jsonLookupTable)

    # close file
    f.close()


def checkExtrinsic():
    for i in range(4):
        camFolder = "cam" + str(i + 1)
        os.chdir(os.path.join("data", camFolder))

        with np.load('camera_matrix.npz') as file:
            intrinsicMatrix, dist = [file[i] for i in ['mtx', 'dist']]
        with np.load('camera_matrix_extrinsic.npz') as file:
            rotation, translation = [file[i] for i in ['rvec', 'tvec']]

        print("Intrinsic for", camFolder, "is: ", "\n", intrinsicMatrix)
        # print("Dist for", camFolder, "is: ","\n", dist)
        # print("Rotation for", camFolder, "is: ","\n", rotation)
        # print("Translation for", camFolder, "is: ","\n", translation)


        os.chdir("..")
        os.chdir("..")


def GenerateForeground():

    foregroundImages = []
    for i in range(4):
        camFolder = "cam" + str(i + 1)
        path = os.path.join("data", camFolder)
        videoName = os.path.join(path, "video.avi")
        video = cv.VideoCapture(videoName)
        video.set(cv.CAP_PROP_POS_FRAMES, 1)
        ret, frame = video.read()
        result = bs.backgroundSubtraction(frame, i)
        cv.imshow("result", result)
        cv.waitKey(0)
        #cv.DestroyAllWindows()
        foregroundImages.append(result)
    return foregroundImages


if __name__ == '__main__':

    #findCameraIntrinsic()
    #findCameraExtrinsic()
    #createLookupTable()
    #checkExtrinsic()
    #checkVoxels()
    backgroundModels = bs.createBackgroundModel()
    GenerateForeground()
    #bs.backgroundSubtraction(backgroundModels)
