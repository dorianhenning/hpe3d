# This script is borrowed and extended from https://github.com/nkolot/SPIN/blob/master/constants.py
# Adhere to their licence to use this script

# Constants
FOCAL_LENGTH = [384.632623, 384.853489]
IMG_RES = 224
IMG_SIZE = [640, 480]
CAMERA_CENTER = [325.821604, 236.576519]
DISTORTION = [0.007027, -0.002393, -0.001758, 0.001477]

# Camera 0 intrinsics
K_R = [[FOCAL_LENGTH[0], 0.0, CAMERA_CENTER[0]],
       [0.0, FOCAL_LENGTH[1], CAMERA_CENTER[1]],
       [0.0, 0.0, 1.0]]

# Camera 0 extrinsics
T_SC = [[0.999996, 0.002800, -0.000146, -0.004579],
        [-0.002800, 0.999992, -0.002714, 0.002799],
        [0.000139, 0.002714, 0.999996, 0.010604],
        [0.000000, 0.000000, 0.000000, 1.000000]]

# Camera 1 extrinsics
T_SC1 = [[0.999994, 0.002767, -0.002155, 0.045409],
         [-0.002772, 0.999993, -0.002392, 0.002625],
         [0.002148, 0.002398, 0.999995, 0.010866],
         [0.000000, 0.000000, 0.000000, 1.000000]]

# Camera 0 to Model (VICON)
T_MC = [[0.999762, 0.021810, 0.000856, -0.030266],
        [-0.021807, 0.999756, -0.003479, 0.000630],
        [-0.000932, 0.003459, 0.999994, 0.039923],
        [0.000000, 0.000000, 0.000000, 1.000000]]

# Mean and standard deviation for normalizing input image
IMG_NORM_MEAN = [0.485, 0.456, 0.406]
IMG_NORM_STD = [0.229, 0.224, 0.225]


MHAD_JOINT_NAMES = [
    'Hips',
    'spine',
    'spine1',
    'spine2',
    'Neck',
    'Head',
    'RightShoulder',
    'RightArm',
    'RightArmRoll',
    'RightForeArm',
    'RightForeArmRoll',
    'RightHand',
    'RightUpLeg',
    'RightUpLegRoll',
    'RightLeg',
    'RightLegRoll',
    'RightFoot',
    'RightToeBase',
    'LeftShoulder',
    'LeftArm',
    'LeftArmRoll',
    'LeftForeArm',
    'LeftForeArmRoll',
    'LeftHand',
    'LeftUpLeg',
    'LeftUpLegRoll',
    'LeftLeg',
    'LeftLegRoll',
    'LeftFoot',
    'LeftToeBase'
]

MHAD_2_HPE3D = [0, 2, 4, 5, 7, 10, 11, 12, 14, 16, 17, 19, 22, 23, 24, 26, 28, 29]
SPIN_2_HPE3D = [8, 41, 1, 42, 2, 3, 4, 9, 10, 11, 22, 5, 6, 7, 12, 13, 14, 19]
HPE3D_2_3DPW = range(25)

JOINT_MAPS = {'mhad': MHAD_2_HPE3D, 'spin': SPIN_2_HPE3D, '3dpw': HPE3D_2_3DPW}

SPIN_JOINT_NAMES = [
    # 25 OpenPose joints (in the order provided by OpenPose)
    'OP Nose',
    'OP Neck',
    'OP RShoulder',
    'OP RElbow',
    'OP RWrist',
    'OP LShoulder',
    'OP LElbow',
    'OP LWrist',
    'OP MidHip',
    'OP RHip',
    'OP RKnee',
    'OP RAnkle',
    'OP LHip',
    'OP LKnee',
    'OP LAnkle',
    'OP REye',
    'OP LEye',
    'OP REar',
    'OP LEar',
    'OP LBigToe',
    'OP LSmallToe',
    'OP LHeel',
    'OP RBigToe',
    'OP RSmallToe',
    'OP RHeel',
    # 24 Ground Truth joints (superset of joints from different datasets)
    'Right Ankle',
    'Right Knee',
    'Right Hip',
    'Left Hip',
    'Left Knee',
    'Left Ankle',
    'Right Wrist',
    'Right Elbow',
    'Right Shoulder',
    'Left Shoulder',
    'Left Elbow',
    'Left Wrist',
    'Neck (LSP)',
    'Top of Head (LSP)',
    'Pelvis (MPII)',
    'Thorax (MPII)',
    'Spine (H36M)',
    'Jaw (H36M)',
    'Head (H36M)',
    'Nose',
    'Left Eye',
    'Right Eye',
    'Left Ear',
    'Right Ear'
]

# Map joints to SMPL joints
SPIN_JOINT_MAP = {
    'OP Nose': 24, 'OP Neck': 12, 'OP RShoulder': 17,
    'OP RElbow': 19, 'OP RWrist': 21, 'OP LShoulder': 16,
    'OP LElbow': 18, 'OP LWrist': 20, 'OP MidHip': 0,
    'OP RHip': 2, 'OP RKnee': 5, 'OP RAnkle': 8,
    'OP LHip': 1, 'OP LKnee': 4, 'OP LAnkle': 7,
    'OP REye': 25, 'OP LEye': 26, 'OP REar': 27,
    'OP LEar': 28, 'OP LBigToe': 29, 'OP LSmallToe': 30,
    'OP LHeel': 31, 'OP RBigToe': 32, 'OP RSmallToe': 33, 'OP RHeel': 34,
    'Right Ankle': 8, 'Right Knee': 5, 'Right Hip': 45,
    'Left Hip': 46, 'Left Knee': 4, 'Left Ankle': 7,
    'Right Wrist': 21, 'Right Elbow': 19, 'Right Shoulder': 17,
    'Left Shoulder': 16, 'Left Elbow': 18, 'Left Wrist': 20,
    'Neck (LSP)': 47, 'Top of Head (LSP)': 48,
    'Pelvis (MPII)': 49, 'Thorax (MPII)': 50,
    'Spine (H36M)': 51, 'Jaw (H36M)': 52,
    'Head (H36M)': 53, 'Nose': 24, 'Left Eye': 26,
    'Right Eye': 25, 'Left Ear': 28, 'Right Ear': 27
}

# Joint selectors
# Indices to get the 14 LSP joints from the 17 H36M joints
H36M_TO_J17 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9]
H36M_TO_J14 = H36M_TO_J17[:14]
# Indices to get the 14 LSP joints from the ground truth joints
J24_TO_J17 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 14, 16, 17]
J24_TO_J14 = J24_TO_J17[:14]
