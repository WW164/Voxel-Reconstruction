import cv2 as cv
import os
import random
import numpy as np
import background_subtraction as bs
import pickle

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
    cv.drawChessboardCorners(image, (CellWidth, CellHeight), corners2, True)
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


# Find corners for a given image
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
        cv.drawChessboardCorners(sampleImage, (CellWidth, CellHeight), corners2, ret)
        cv.imshow('img', sampleImage)
        cv.waitKey(50)
    # If fail to find the corners automatically wait for user manual input
    if not ret:
        cv.imshow('img', sampleImage)
        cv.setMouseCallback('img', click_event, param=sampleImage)
        while not manual_points_entered:
            cv.imshow('img', sampleImage)
            cv.waitKey(500)

    return objPoints, imgPoints, ret


# Find camera matrix by finding corners of a chessboard and calibrateCamera function
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
        os.chdir("..")
        os.chdir("..")


# Draw axis for a given set of image points and origin
def draw(image, corners, imgPts):
    corner = tuple(corners[0].ravel().astype(int))
    img = cv.line(image, corner, tuple(imgPts[0].ravel().astype(int)), (255, 0, 0), 5)
    img = cv.line(image, corner, tuple(imgPts[1].ravel().astype(int)), (0, 255, 0), 5)
    img = cv.line(image, corner, tuple(imgPts[2].ravel().astype(int)), (0, 0, 255), 5)
    cv.imshow("img", img)
    cv.waitKey(0)


# Find camera extrinsic parameters by finding corners and solvePnP function
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
        # Draw axis for confirmation
        imgPts, jac = cv.projectPoints(axis, rotation, translation, intrinsicMatrix, dist)
        draw(image, corners2, imgPts)

        os.chdir("..")
        os.chdir("..")

    cv.destroyAllWindows()


def GetForeground(camera):
    camFolder = "cam" + str(camera)
    path = os.path.join("data", camFolder, 'foreground.png')
    img = cv.imread(path)
    dimensions = img.shape
    for x in range(0, dimensions[0]):
        for y in range(0, dimensions[1]):
            if np.linalg.norm(img[x, y]) > 1:
                print(x, y)


# Create lookup table by iterating through voxel space and projecting points
def checkVoxels():
    cameraLookupTable = {}

    # Define the range of the cube
    Xl = 0
    Xh = 7
    Yl = -4
    Yh = 6
    Zl = 2
    Zh = -13

    # save the output in result
    result = []

    for i in range(4):
        # Read in the foreground image.
        voxelCoordinates = []
        camFolder = "cam" + str(i + 1)
        path = os.path.join("data", camFolder)
        foregroundImagePath = os.path.join("data", camFolder, 'foreground.png')
        foregroundImage = cv.imread(foregroundImagePath)

        # read in the camera matrix
        with np.load(os.path.join(path, 'camera_matrix.npz')) as file:
            intrinsicMatrix, dist = [file[i] for i in ['mtx', 'dist']]
        with np.load(os.path.join(path, 'camera_matrix_extrinsic.npz')) as file:
            rotation, translation = [file[i] for i in ['rvec', 'tvec']]

        for x in np.arange(Xl, Xh, 0.25):
            for y in np.arange(Yl, Yh, 0.25):
                for z in np.arange(Zh, Zl, 0.25):

                    output = []
                    # Get the projected point of the voxel position.
                    voxelPoint = np.float32((x, y, z)) * tileSize
                    voxelCoordinate, jac = cv.projectPoints(voxelPoint, rotation, translation, intrinsicMatrix, dist)
                    fx = int(voxelCoordinate[0][0][0])
                    fy = int(voxelCoordinate[0][0][1])

                    Xc = voxelPoint[0]
                    Yc = voxelPoint[1]
                    Zc = voxelPoint[2]

                    output.append((fy, fx))
                    voxelCoordinates.append(voxelCoordinate)
                    # Store 3d points as key and image points as value
                    if (Xc, Yc, Zc) in cameraLookupTable:
                        cameraLookupTable[(Xc, Yc, Zc)].append((fy, fx))
                    else:
                        cameraLookupTable[(Xc, Yc, Zc)] = output


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
        # cv.DestroyAllWindows()
        foregroundImages.append(result)
    return foregroundImages


# Generate lookup table for XOR method
def xorLookupTable():
    cameraLookupTable = {}

    # Define the range of the cube
    Xl = 0
    Xh = 7
    Yl = -4
    Yh = 6
    Zl = 2
    Zh = -13

    for i in range(4):

        voxelCoordinates = []

        # Read in the foreground image.
        camFolder = "cam" + str(i + 1)
        path = os.path.join("data", camFolder)
        videoName = os.path.join(path, "video.avi")
        video = cv.VideoCapture(videoName)
        success, frame = video.read()

        # read in the camera matrix
        with np.load(os.path.join(path, 'camera_matrix.npz')) as file:
            intrinsicMatrix, dist = [file[i] for i in ['mtx', 'dist']]
        with np.load(os.path.join(path, 'camera_matrix_extrinsic.npz')) as file:
            rotation, translation = [file[i] for i in ['rvec', 'tvec']]

        for x in np.arange(Xl, Xh, 0.5):
            for y in np.arange(Yl, Yh, 0.5):
                for z in np.arange(Zh, Zl, 0.5):
                    # Get the projected point of the voxel position.
                    voxelPoint = np.float32((x, y, z)) * tileSize
                    voxelCoordinate, jac = cv.projectPoints(voxelPoint, rotation, translation, intrinsicMatrix, dist)
                    fx = int(voxelCoordinate[0][0][0])
                    fy = int(voxelCoordinate[0][0][1])

                    Xc = voxelPoint[0]
                    Yc = voxelPoint[1]
                    Zc = voxelPoint[2]
                    # Store 2d points as key and array of voxels as value
                    if (fy, fx) in cameraLookupTable:
                        cameraLookupTable[(fy, fx)].append((Xc, Yc, Zc, i))
                    else:
                        cameraLookupTable[(fy, fx)] = [(Xc, Yc, Zc, i)]

    with open('xor.pickle', 'wb') as handle:
        pickle.dump(cameraLookupTable, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(cameraLookupTable)


# Store colors for each pixel
def checkColor():
    bgr = {}

    for i in range(4):

        camFolder = "cam" + str(i + 1)
        path = os.path.join("data", camFolder)
        videoName = os.path.join(path, "video.avi")
        video = cv.VideoCapture(videoName)

        with np.load(os.path.join(path, 'camera_matrix.npz')) as file:
            intrinsicMatrix, dist = [file[i] for i in ['mtx', 'dist']]

        success, frame = video.read()
        if success:
            height, width = frame.shape[: 2]

        # Loop through every pixel in the image
        for y in range(height):
            for x in range(width):
                # Find distance to each camera using camera matrix and projecting points to 3d camera coordinate system
                imagePoint = np.array([x, y, 1]).reshape(3, 1)
                cameraOrigin = np.array([0, 0, 0]).reshape(3, 1)
                inverseCameraMatrix = np.linalg.inv(intrinsicMatrix)
                imagePointCameraCoord = np.dot(inverseCameraMatrix, imagePoint)

                distance = np.sqrt(np.sum((imagePointCameraCoord - cameraOrigin) ** 2, axis=0))

                # Get the RGB color of the pixel
                (blue, green, red) = frame[y, x]
                if (x, y) in bgr:
                    bgr[(x, y)].append((blue, green, red, i, tuple(distance)))
                else:
                    bgr[(x, y)] = [(blue, green, red, i, tuple(distance))]

        print("Done")


    # with open('test.pickle', 'wb') as handle:
    #     pickle.dump(bgr, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('test.pickle', 'rb') as handle:
        bgr = pickle.load(handle)

    colorData = {}
    for point in bgr:
        distances = []

        for color in bgr[point]:
            distances.append(color[4])
        # Find the minimum distance of each pixel from cameras and remove other colors
        minDistance = distances.index(min(distances))
        for color in bgr[point]:
            if color[3] == minDistance:
                colorData[point] = (color[0], color[1], color[2])
    # with open('colorTest.pickle', 'wb') as handle:
    #     pickle.dump(colorData, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(colorData)


#if __name__ == '__main__':
    # findCameraIntrinsic()
    # findCameraExtrinsic()
    # createLookupTable()
    #checkVoxels()
    # xorLookupTable()

    # checkColor()
    # xorLookupTable()
    # backgroundModels = bs.createBackgroundModel()
    # GenerateForeground()
    # bs.backgroundSubtraction(backgroundModels)
