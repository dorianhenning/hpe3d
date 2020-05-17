import numpy as np
import os
import cv2
import pickle as pkl
import torch
from tqdm import tqdm

# Detection imports
from hpe3d.models import hmr
from hpe3d.utils.img_utils import FakeCamera
from hpe3d.utils.kp_utils import bbox_from_kp2d
import hpe3d.utils.config as cfg


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

spin = hmr()

dataset_path = cfg.ROOT_3DPW


def main(data_file):

    _, clip_name = os.path.split(data_file)
    seq = pkl.load(open(os.path.join(dataset_path, 'sequenceFiles', data_file + '.pkl'), 'rb'), encoding='latin1')
    if len(seq['genders']) > 1:
        raise NotImplementedError('Currently only sequences with one model!')
        return

    ds_length = len(seq['img_frame_ids'])

    rel_img_path = os.path.join('imageFiles', clip_name)
    first_image = cv2.imread(os.path.join(dataset_path, rel_img_path, 'image_00000.jpg'))

    # save information in dictionary:
    pred_betas = []
    pred_poses = []
    pred_rotmat = []
    pred_camera = []
    gt_poses = seq['poses'][0]
    gt_betas = seq['betas'][0]
    gt_bboxes = []
    gt_kp2d = seq['poses2d'][0]
    gt_kp3d = seq['jointPositions']
    gt_T_CW = seq['cam_poses']
    gt_K = seq['cam_intrinsics']
    image_size = list(first_image.shape[:-1])
    image_names = []
    depth_names = []

    for i in tqdm(range(ds_length)):
        image_name = 'image_%05i.jpg' % i

        full_image = cv2.imread(os.path.join(dataset_path, rel_img_path, image_name))

        # GT bounding box:
        img_size = list(full_image.shape[:-1])
        gt_bbox = [bbox_from_kp2d(gt_kp2d[i, :-1, :-1].T, img_size)]  # list of bboxes, batch_size = 1

        cam = FakeCamera(gt_bbox, K_r=gt_K, img_size=img_size)

        det, norm = cam.preprocess_image(full_image[np.newaxis])  # new axis for batch size

        rotmat, poses, betas, camera = spin(norm)

        pred_betas.append(betas)
        pred_poses.append(poses)
        pred_rotmat.append(rotmat)
        pred_camera.append(camera)
        gt_bboxes.append(gt_bbox)
        image_names.append(os.path.join(rel_img_path, image_name))

    seq_pred = dict([
        ('pred_betas', pred_betas),
        ('pred_poses', pred_poses),
        ('pred_rotmats', pred_rotmat),
        ('pred_cameras', pred_camera),
        ('gt_poses', gt_poses),
        ('gt_betas', gt_betas),
        ('gt_bboxes', gt_bboxes),
        ('gt_kp2d', gt_kp2d),
        ('gt_kp3d', gt_kp3d),
        ('gt_T_CW', gt_T_CW),
        ('gt_K', gt_K),
        ('image_size', image_size),
        ('image_names', image_names),
        ('depth_names', depth_names)
    ])

    pred_folder = os.path.join(cfg.ROOT_HPE3D, 'data', 'Predictions')
    try:
        os.mkdir(pred_folder)
    except FileExistsError:
        pass
    ds_name = '3dpw'
    fname = 'pred_%s_%s.pkl' % (ds_name, clip_name)
    with open(os.path.join(pred_folder, fname), 'wb') as handle:
        pkl.dump(seq_pred, handle)


if __name__ == "__main__":

    data_folder = os.path.join(dataset_path, 'sequenceFiles')
    data_files = []
    for root, _, files in os.walk(data_folder):
        for f in files:
            data_files.append(os.path.join(root, f.rstrip('.pkl')))

    for data_file in data_files:
        print(data_file)
        main(data_file)
