import numpy as np
import torch
from torch.nn import functional as F


def rot6d_to_rotmat(x):
    '''
        Convert 6D rotation representation to 3x3 rotation matrix.
        Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
        Input:
            (B,6) Batch of 6-D rotation representations
        Output:
            (B,3,3) Batch of corresponding rotation matrices
    '''
    x = x.view(-1, 3, 2)
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1)


def rotation(rad, axis):
    '''
        Returns the rotation matrix in 3D (R \in SO(3)) around the given axis (x,y,z)
        input param:
            deg:        radians or rotation
            axis:       ['x', 'y', 'z']
        return:
            R:          rotation matrix in SO(3)
    '''
    cs = np.cos(float(rad))
    sn = np.sin(float(rad))

    if axis == 'x':
        return np.array([[1., 0., 0.], [0., cs, -sn], [0., sn, cs]], dtype=float)
    elif axis == 'y':
        return np.array([[cs, 0., sn], [0., 1., 0.], [-sn, 0., cs]], dtype=float)
    elif axis == 'z':
        return np.array([[cs, -sn, 0.], [sn, cs, 0.], [0., 0., 1.]], dtype=float)
    else:
        raise ValueError('unknown axis')


def quat2rot(quat):
    '''
        Computes the rotation matrix given the
        quaternion:
        input param:
            quat:       4D quaternion
        return:
            R:          3 x 3 rotation matrix
    '''
    s = 1 / (np.linalg.norm(quat) * np.linalg.norm(quat))

    R = np.eye(3)
    R[0, 0] += -2 * s * (quat[1] * quat[1] + quat[2] * quat[2])
    R[1, 1] += -2 * s * (quat[0] * quat[0] + quat[2] * quat[2])
    R[2, 2] += -2 * s * (quat[1] * quat[1] + quat[0] * quat[0])
    R[1, 0] = 2 * s * (quat[0] * quat[1] + quat[2] * quat[3])
    R[0, 1] = 2 * s * (quat[0] * quat[1] - quat[2] * quat[3])
    R[2, 0] = 2 * s * (quat[0] * quat[2] - quat[1] * quat[3])
    R[0, 2] = 2 * s * (quat[0] * quat[2] + quat[1] * quat[3])
    R[2, 1] = 2 * s * (quat[1] * quat[2] + quat[0] * quat[3])
    R[1, 2] = 2 * s * (quat[1] * quat[2] - quat[0] * quat[3])

    return R


def rot2quat(rot):
    '''
        Computes the quaternion given the
        rotation matrix:
        input param:
            R:          3 x 3 rotation matrix
        return:
            quat:       4D quaternion
    '''
    tr = np.trace(rot)

    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (rot[2, 1] - rot[1, 2]) / S
        qy = (rot[0, 2] - rot[2, 0]) / S
        qz = (rot[1, 0] - rot[0, 1]) / S
    elif ((rot[0, 0] > rot[1, 1]) & (rot[0, 0] > rot[2, 2])):
        S = np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2
        qw = (rot[2, 1] - rot[1, 2]) / S
        qx = 0.25 * S
        qy = (rot[0, 1] + rot[1, 0]) / S
        qz = (rot[0, 2] + rot[2, 0]) / S
    elif (rot[1, 1] > rot[2, 2]):
        S = np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2
        qw = (rot[0, 2] - rot[2, 0]) / S
        qx = (rot[0, 1] + rot[1, 0]) / S
        qy = 0.25 * S
        qz = (rot[1, 2] + rot[2, 1]) / S
    else:
        S = np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2
        qw = (rot[1, 0] - rot[0, 1]) / S
        qx = (rot[0, 2] + rot[2, 0]) / S
        qy = (rot[1, 2] + rot[2, 1]) / S
        qz = 0.25 * S

    return np.array([qx, qy, qz, qw], dtype=np.float32)


def center2corner(center, scale):
    '''
        changes the center/scale representation of the bounding box
        to the one used as fake camera parameter
        input param:
            center:     center of the bounding box in uv coordinates
            scale:      relative scale of the bounding box (to the norm size)
        return:
            bbox:       list of corner coordinates of bounding box
    '''
    h = scale * 200. / .9
    ul = center[0] - h / 2
    vt = center[1] - h / 2
    ur = center[0] + h / 2
    vb = center[1] + h / 2
    return [ul, vt, ur, vb]


def convert_smpl_joint_hom(joint):

    j = joint.copy()
    if len(joint.shape) == 1:
        j = np.insert(j, 3, 1)
        j[1] *= -1.
        j[2] *= -1.
    elif len(joint.shape) == 2:
        j = np.insert(j, 3, 1, axis=1)
        j[:, 1] *= -1.
        j[:, 2] *= -1.
    else:
        j = np.insert(j, 3, 1, axis=2)
        j[:, :, 1] *= -1.
        j[:, :, 2] *= -1.

    return j
