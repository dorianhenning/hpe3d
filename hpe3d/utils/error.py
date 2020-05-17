import numpy as np


def mpjpe(j_pred, j_gt, align=True):
    '''
        Computes the mean per joint position error
        with or without alignment:
        input param:
            j_pred:     B x N x 3 predicted batch of joints
            j_gt:       B x N x 3 grourd truth
            align:      boolean if to align joints to pelvis (N = 0)
        return:
            mpjpe:      in [m]
    '''

    if align:
        j_gt = j_gt - j_gt[:, 0:1]
        j_pred = j_pred - j_pred[:, 0:1]

    mpjpe = np.mean(np.linalg.norm((j_pred - j_gt), axis=2))

    return mpjpe
