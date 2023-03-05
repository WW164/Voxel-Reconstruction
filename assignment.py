import glm
import random
import numpy as np
import cv2 as cv
import os
from numpy import load
import pickle
import background_subtraction as bs
import time

block_size = 1
frameCellWidth = 1000
frameCellHeight = 1000
tileSize = 115
frameIndex = 1
pixels = []
voxels = []
previousForegroundImages = []
voxelsOnCam = {0: [], 1: [], 2: [], 3: []}


# Load the lookup table.
def loadPickle(type):
    if type == 'voxels':
        with open('lookuptable.pickle', 'rb') as handle:
            lookupTable = pickle.load(handle)
    else:
        with open('xor.pickle', 'rb') as handle:
            lookupTable = pickle.load(handle)

    return lookupTable


# Load the color file.
def loadColor():
    with open('colorTest.pickle', 'rb') as handle:
        colorData = pickle.load(handle)

    return colorData


# Load in the cameras intrinsic and extrinsic data
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


# Load in the current frame and pass it to the background subtraction algorithm.
# Returns the 4 foreground images for this frame
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


# Determine which voxels are "on" in all four cameras then find the colour for that voxel.
def finaliseVoxels(width, height, depth):
    global voxels

    colorFlag = False

    data, colors = [], []
    onVoxels = set.intersection(set(voxelsOnCam[0]), set(voxelsOnCam[1]), set(voxelsOnCam[2]), set(voxelsOnCam[3]))

    temp = 0

    for vox in onVoxels:

        Xc = vox[0]
        Yc = vox[1]
        Zc = vox[2]

        scalar = 0.01
        fixedPoint = (Xc * scalar, -Zc * scalar, Yc * scalar)
        data.append((fixedPoint[0],
                     fixedPoint[1],
                     fixedPoint[2]))

        if colorFlag:
            colorData = loadColor()
            flag = True
            for pixel in pixels:
                for coordinates in pixels[pixel]:
                    # For each on-voxel finds the voxel and its corresponding pixel
                    if (Xc, Yc, Zc) == (coordinates[0], coordinates[1], coordinates[2]):
                        temp += 1
                        while flag:
                            if pixel in colorData:
                                # Append the pixel color to colors
                                blue = (colorData[pixel][0]) / 255
                                green = (colorData[pixel][1]) / 255
                                red = (colorData[pixel][2]) / 255
                                colors.append([blue, green, red])
                                flag = False
        else:
            colors.append([Xc / width, Zc / depth, Yc / height])

    return data, colors


# Loop through every pixel in the lookup table and determine if its on, if so add to camera voxel mapping.
def FirstFrameVoxelPositions(foregroundImages, width, height, depth):
    global voxelsOnCam

    for pixel in pixels:
        for j in range(len(foregroundImages)):
            if np.linalg.norm(foregroundImages[j][pixel]) > 1:
                for voxel in pixels[pixel]:
                    if voxel[3] == j:
                        vCoord = (voxel[0], voxel[1], voxel[2])
                        voxelsOnCam[j].append(vCoord)
    data, colors = finaliseVoxels(width, height, depth)

    return data, colors


# Loop through changed pixels and add or remove them from the camera voxel mapping respectively
def XORFrameVoxelPositions(currImgs, prevImgs, width, height, depth):
    global voxelsOnCam, pixels
    start_time = time.time()
    for i in range(len(currImgs)):
        mask_xor = cv.bitwise_xor(currImgs[i], prevImgs[i])
        nowOnPixels = cv.bitwise_and(currImgs[i], mask_xor)
        nowOnPixels = np.argwhere(nowOnPixels == 255)
        for pixel in nowOnPixels:
            x, y = pixel
            if (x, y) in pixels:
                for values in pixels[(x, y)]:
                    if values[3] == i:
                        vCoord = (values[0], values[1], values[3])
                        voxelsOnCam[i].append(vCoord)

        not_current_image = (255-currImgs[i])
        nowOffPixels = cv.bitwise_and(not_current_image, mask_xor)
        nowOffPixels = np.argwhere(nowOffPixels == 255)
        for pixel in nowOffPixels:
            x, y = pixel
            if (x, y) in pixels:
                for values in pixels[(x, y)]:
                    if values[3] == i:
                        vCoord = (values[0], values[1], values[3])
                        for j in range(len(voxelsOnCam[i]) - 1):
                            if voxelsOnCam[i][j] == vCoord:
                                del voxelsOnCam[i][j]
                                break

    data, colors = finaliseVoxels(width, height, depth)
    print("XOR Method took ", time.time() - start_time, "to run")
    return data, colors


# Run the appropriate function based on the current frame. Save previous foreground images.
def set_voxel_positions(width, height, depth):
    global frameIndex, previousForegroundImages
    foregroundImages = GenerateForeground()

    if frameIndex == 1:
        data, colors = FirstFrameVoxelPositions(foregroundImages, width, height, depth)

    else:
        data, colors = (XORFrameVoxelPositions(foregroundImages, previousForegroundImages, width, height, depth))
    previousForegroundImages = foregroundImages
    frameIndex += 1
    return data, colors


# Load relevant data and then set the cameras position based on its extrinsic data.
def get_cam_positions():
    # loading lookup table from json file
    global pixels, voxels
    pixels = loadPickle("xor")
    voxels = loadPickle("voxels")
    bs.createBackgroundModel()

    rvecs, tvecs, intrinsicMatrix, dist = getData()
    Positions = []
    for i in range(4):
        rotM = cv.Rodrigues(rvecs[i])[0]
        camPos = -rotM.T.dot(tvecs[i])
        camPosFix = [camPos[0], -camPos[2], camPos[1]]
        Positions.append(camPosFix)

    return [Positions[0], Positions[1], Positions[2], Positions[3]], \
        [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [1.0, 1.0, 0]]


# set the cameras rotations based on its extrinsic data.
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

    # print("rotation cam1:", "\n", RotMs[0], "\n",
    #      "rotation cam2:", "\n", RotMs[1], "\n",
    #      "rotation cam3:", "\n", RotMs[2], "\n",
    #      "rotation cam4:", "\n", RotMs[3], "\n",)

    for c in range(len(cam_rotations)):
        cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][0] * np.pi / 180, [1, 0, 0])
        cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][1] * np.pi / 180, [0, 1, 0])
        cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][2] * np.pi / 180, [0, 0, 1])
    return cam_rotations
