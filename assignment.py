import glm
import random
import numpy as np
import cv2 as cv
import os
from numpy import load
import json
import background_subtraction as bs


block_size = 1
frameCellWidth = 1000
frameCellHeight = 1000
tileSize = 115
frameIndex = 1
voxels = []


def loadJson():
    # Opening JSON file
    f = open('dict.json')

    # returns JSON object as
    # a dictionary
    lookupTable = json.load(f)

    # Iterating through the json
    # list
    #for voxel in lookupTable:
        #print(voxel, ":", lookupTable[voxel])

    # Closing file
    f.close()

    return lookupTable


def getData():
    rvecs = []
    tvecs = []
    intrinsicMatrix = []
    dist = []
    for i in range(4):
        camFolder = "cam" + str(i + 1)
        data = load('data/' + camFolder + '/camera_matrix_extrinsic.npz')
        lst = data.files
        for item in lst:
            if item == 'rvec':
                rvecs.append(data[item])
            if item == 'tvec':
                tvecs.append(np.divide(data[item], tileSize))

        data = load('data/' + camFolder + '/camera_matrix.npz')
        lst = data.files
        for item in lst:
            if item == 'mtx':
                intrinsicMatrix.append(data[item])
            if item == 'dist':
                dist.append(data[item])

    return rvecs, tvecs, intrinsicMatrix, dist


def generate_grid(width, depth):
    # Generates the floor grid locations
    # You don't need to edit this function
    data, colors = [], []
    for x in range(width):
        for z in range(depth):
            data.append([x * block_size - width / 2, -block_size, z * block_size - depth / 2])
            colors.append([1.0, 1.0, 1.0] if (x + z) % 2 == 0 else [0, 0, 0])
    return data, colors


def GetForegroundValue(foregroundImages, index, coords):
    coords = coords.replace('(', '')
    coords = coords.replace(')', '')
    coords = coords.replace(' ', '')

    x, y = coords.split(',', -1)

    img = foregroundImages[index]

    #temp = cv.circle(img, (int(x), int(y)), 5, (255, 0, 0), 2)
    #print([int(x), int(y)])
    #cv.imshow('temp', temp)
    #cv.waitKey(0)
    return np.linalg.norm(img[int(x), int(y)]) > 1


def GenerateForeground():
    global frameIndex
    foregroundImages = []
    for i in range(4):
        camFolder = "cam" + str(i + 1)
        path = os.path.join("data", camFolder)
        videoName = os.path.join(path, "video.avi")
        video = cv.VideoCapture(videoName)
        totalFrames = video.get(cv.CAP_PROP_FRAME_COUNT)
        # check for valid frame number
        if frameIndex >= 0 & frameIndex <= totalFrames:
            video.set(cv.CAP_PROP_POS_FRAMES, frameIndex)
            ret, frame = video.read()
            result = bs.backgroundSubtraction(frame, i)
            foregroundImages.append(result)
        else:
            print("ERROR: Invalid Frame")

    return foregroundImages


def set_voxel_positions(width, height, depth):
    global frameIndex
    # loading lookup table from json file
    # TODO: Voxels is a dictionary that each voxel (3d point) is the key and projected 2d points are value
    foregroundImages = GenerateForeground()
    if frameIndex == 1:
        print("first frame")

    else:
        print(frameIndex)

    Xl = 0
    Xh = 7
    Yl = -4
    Yh = 5
    Zl = 2
    Zh = -13

    data, colors = [], []

    for x in range(Xl, Xh):
        for y in range(Yl, Yh):
            for z in range(Zh, Zl):
                voxelPoint = np.float32((x, y, z)) * tileSize
                Xc = voxelPoint[0]
                Yc = voxelPoint[1]
                Zc = voxelPoint[2]
                boolValues = []
                for j in range(4):
                    if GetForegroundValue(foregroundImages, j, voxels[str((Xc, Yc, Zc))][j]):
                        boolValues.append(True)
                    else:
                        boolValues.append(False)
                        break

                # #print(boolValues)
                scalar = 0.01
                fixedPoint = (Xc * scalar, -Zc * scalar, Yc * scalar)

                if np.all(boolValues):
                    data.append((fixedPoint[0],
                                 fixedPoint[1],
                                 fixedPoint[2]))
                    colors.append([x / width, z / depth, y / height])

    frameIndex += 1
    return data, colors


def get_cam_positions():
    rvecs, tvecs, intrinsicMatrix, dist = getData()
    global voxels
    voxels = loadJson()
    bs.createBackgroundModel()
    Positions = []
    for i in range(4):
        rotM = cv.Rodrigues(rvecs[i])[0]
        camPos = -rotM.T.dot(tvecs[i])
        camPosFix = [camPos[0], -camPos[2], camPos[1]]
        Positions.append(camPosFix)

    return [Positions[0], Positions[1], Positions[2], Positions[3]], \
           [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [1.0, 1.0, 0]]


def get_cam_rotation_matrices():
    rvecs, tvecs, intrinsicMatrix, dist = getData()
    RotMs = []
    for i in range(4):
        rvec = np.array((rvecs[i][0], rvecs[i][1], rvecs[i][2]))
        rotM = cv.Rodrigues(rvec)[0]
        rotM1 = np.identity(4)
        rotM1[:3, :3] = rotM
        RotMs.append(rotM1)

    cam_angles = [[0, 0, 90], [0, 0, 90], [0, 0, 90], [0, 0, 90]]
    cam_rotations = [glm.mat4(RotMs[0]), glm.mat4(RotMs[1]), glm.mat4(RotMs[2]), glm.mat4(RotMs[3])]

    for c in range(len(cam_rotations)):
        cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][0] * np.pi / 180, [1, 0, 0])
        cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][1] * np.pi / 180, [0, 1, 0])
        cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][2] * np.pi / 180, [0, 0, 1])
    return cam_rotations
