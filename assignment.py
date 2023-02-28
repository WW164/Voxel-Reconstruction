import glm
import random
import numpy as np
import cv2 as cv
from numpy import load


block_size = 1.0
frameCellWidth = 1000
frameCellHeight = 1000

def getData():
    rvecs = []
    tvecs = []
    for i in range(4):
        camFolder = "cam" + str(i + 1)
        data = load('data/' + camFolder + '/camera_matrix_extrinsic.npz')
        lst = data.files
        for item in lst:
            if item == 'rvec':
                rvecs.append(data[item])
            if item == 'tvec':
                tvecs.append(data[item])
    return rvecs, tvecs


def generate_grid(width, depth):
    # Generates the floor grid locations
    # You don't need to edit this function
    data, colors = [], []
    for x in range(width):
        for z in range(depth):
            data.append([x*block_size - width/2, -block_size, z*block_size - depth/2])
            colors.append([1.0, 1.0, 1.0] if (x+z) % 2 == 0 else [0, 0, 0])
    return data, colors


def set_voxel_positions(width, height, depth):

    with np.load('camera_matrix.npz') as file:
        intrinsicMatrix, dist = [file[i] for i in ['mtx', 'dist']]
    with np.load('camera_matrix_extrinsic.npz') as file:
        rotation, translation = [file[i] for i in ['rvec', 'tvec']]

    Xl = int(-width/2)
    Xh = int(width/2)
    Yl = int(-height/2)
    Yh = int(height/2)
    Zl = int(-depth/2)
    Zh = int(depth/2)


    # Generates random voxel locations
    # TODO: You need to calculate proper voxel arrays instead of random ones.
    data, colors = [], []
    for x in range(Xl, Xh, 4):
        for y in range(Yl, Yh, 4):
            for z in range(Zl, Zh, 4):
                voxelPoint = np.float32(x, y, z)
                voxelCoordinates, jac = cv.projectPoints(voxelPoint, rotation, translation, intrinsicMatrix, dist)
                data.append([x, y, z])

                # if random.randint(0, 1000) < 5:
                #     data.append([x*block_size - width/2, y*block_size, z*block_size - depth/2])
                #     colors.append([x / width, z / depth, y / height])

    #data.append(framePoint)

    return data


def get_cam_positions():
    
    rvecs, tvecs = getData()
    Positions = []
    for i in range(4):
        rotM = cv.Rodrigues(rvecs[i])[0]
        camPos = -rotM.T.dot(tvecs[i])
        camPosFix = [camPos[0], -camPos[2], -camPos[1]]
        Positions.append(camPosFix)

    return [Positions[0], Positions[1], Positions[2], Positions[3]], \
        [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [1.0, 1.0, 0]]


def get_cam_rotation_matrices():

    rvecs, tvecs = getData()
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

