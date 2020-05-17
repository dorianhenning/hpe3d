import numpy as np
import json
import hpe3d.utils.constants as C


def read_openpose(json_file, dataset='bmhad'):
    # read the openpose detection
    json_data = json.load(open(json_file, 'r'))
    people = json_data['people']

    if len(people) > 0:
        p_sel = 0
        kp25 = np.reshape(people[p_sel]['pose_keypoints_2d'], [25, 3])
    else:
        kp25 = np.zeros((25, 3), dtype=float)
    return kp25


def get_joints_from_bvh(row):
    joints = np.zeros((30, 3), dtype=float)
    for i, name in enumerate(C.MHAD_JOINT_NAMES):
        for j, ext in enumerate(['.x', '.y', '.z']):
            joints[i, j] = row[name + ext]
    joints *= 10.  # from cm to mm
    return joints


def bbox_from_kp2d(kp2d, img_size):
    max_u = kp2d[:, 0].max()
    min_u = kp2d[:, 0].min()
    max_v = kp2d[:, 1].max()
    min_v = kp2d[:, 1].min()

    w_ = max_u - min_u
    h_ = max_v - min_v

    w2 = w_ * .9  # width is very tight -> broaden more
    h2 = h_ * .6  # height is lengthened more to the top (head joint missing)

    center = [min_u + w_ / 2., min_v + h_ / 2.]
    bbox = [center[0] - w2, center[1] - h2 * 1.2, center[0] + w2, center[1] + h2]
    bbox = np.array(bbox, dtype=int)

    bbox[0] = max(0, min(bbox[0], img_size[0] - 1))
    bbox[2] = max(0, min(bbox[2], img_size[0] - 1))
    bbox[1] = max(0, min(bbox[1], img_size[1] - 1))
    bbox[3] = max(0, min(bbox[3], img_size[1] - 1))

    return bbox
