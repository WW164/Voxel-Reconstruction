import glm
import random
import numpy as np
import cv2 as cv
import os
from numpy import load


block_size = 1.0
frameCellWidth = 1000
frameCellHeight = 1000
tileSize = 115

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
            data.append([x*block_size - width/2, -block_size, z*block_size - depth/2])
            colors.append([1.0, 1.0, 1.0] if (x+z) % 2 == 0 else [0, 0, 0])
    return data, colors


def GetForeground(camera, x, y):
    camFolder = "cam" + str(camera)
    path = os.path.join("data", camFolder, 'foreground.png')
    img = cv.imread(path)

    return img[x, y] > 1


def set_voxel_positions(width, height, depth):

    rotation, translation, intrinsicMatrix, dist = getData()

    Xl = 0
    Xh = 7
    Yl = -4
    Yh = 5
    Zl = 2
    Zh = -13

    # Generates random voxel locations
    # TODO: You need to calculate proper voxel arrays instead of random ones.
    data, colors = [], []
    for i in range(4):
        for x in range(Xl, Xh):
            for y in range(Yl, Yh):
                for z in range(Zh, Zl):
                    voxelPoint = np.float32((x, y, z)) * tileSize
                    voxelCoordinates, jac = cv.projectPoints(voxelPoint, rotation[i], translation[i], intrinsicMatrix[i], dist[i])
                    if GetForeground(i+1, voxelCoordinates[0][0][0], voxelCoordinates[0][0][1]):
                        data.append(voxelCoordinates)
                        colors.append([x / width, z / depth, y / height])
        print("Done")

    #
    #             # if random.randint(0, 1000) < 5:
    #             #     data.append([x*block_size - width/2, y*block_size, z*block_size - depth/2])
    #             #     colors.append([x / width, z / depth, y / height])
    # for x in range(Xl, Xh, 4):
    #     for y in range(Yl, Yh, 4):
    #         for z in range(Zl, Zh, 4):
    #             if random.randint(0, 1000) < 5:
    #                 data.append([x*block_size - width/2, y*block_size, z*block_size - depth/2])
    #                 colors.append([x / width, z / depth, y / height])

    #data.append(framePoint)

    return data, colors


def get_cam_positions():
    
    rvecs, tvecs, intrinsicMatrix, dist = getData()
    Positions = []
    for i in range(4):
        rotM = cv.Rodrigues(rvecs[i])[0]
        camPos = -rotM.T.dot(tvecs[i])
        camPosFix = [camPos[0], -camPos[2], -camPos[1]]
        Positions.append(camPosFix)

    return [Positions[0], Positions[1], Positions[2], Positions[3]], \
        [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [1.0, 1.0, 0]]


def get_cam_rotation_matrices():

    rvecs, tvecs, intrinsicMatrix, dist = getData()
    RotMs = []
    for i in range(4):
        rvec = np.array((rvecs[i][0], -rvecs[i][2], -rvecs[i][1]))
        rotM = cv.Rodrigues(rvec)[0]
        rotM1 = np.identity(4)
        rotM1[:3, :3] = rotM
        RotMs.append(rotM1)

    cam_angles = [[0, 0, -90], [0, 0, -90], [0, 0, -90], [0, 0, -90]]
    cam_rotations = [glm.mat4(RotMs[0]), glm.mat4(RotMs[1]), glm.mat4(RotMs[2]), glm.mat4(RotMs[3])]
    for c in range(len(cam_rotations)):
        cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][0] * np.pi / 180, [1, 0, 0])
        cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][1] * np.pi / 180, [0, 1, 0])
        cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][2] * np.pi / 180, [0, 0, 1])
    return cam_rotations

