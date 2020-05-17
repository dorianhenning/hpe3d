import numpy as np
import os
import cv2
import pickle as pkl
import torch
from tqdm import tqdm
import pandas as pd

# Detection imports
from hpe3d.models import hmr
from hpe3d.utils.img_utils import FakeCamera
from hpe3d.utils.kp_utils import get_joints_from_bvh, bbox_from_kp2d
import hpe3d.utils.config as cfg


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

spin = hmr()

dataset_path = cfg.ROOT_MHAD


def read_calib(cam):

    ext_name = os.path.join(dataset_path, 'Calibration', 'RwTw_%s.txt' % cam)
    int_name = os.path.join(dataset_path, 'Calibration', 'camcfg_%s.yml' % cam)

    with open(ext_name) as fh:
        rw = fh.readline().split('=')[1].rstrip().split(' ')
        tw = fh.readline().split('=')[1].rstrip().split(' ')
        R = np.array(rw).astype(float).reshape(3, 3)
        T = np.array(tw).astype(float)

    fh = cv2.FileStorage(int_name, cv2.FILE_STORAGE_READ)
    K = fh.getNode('Camera_1').getNode('K').mat()

    return R, T, K


def main(data_file):
    info = data_file.split('_')
    _, sub, act, rep = info

    # Kinect k01 is the frontal view kinect camera of the dataset
    kin = 'k01'

    rel_img_path = os.path.join('Kinect', kin, sub.upper(), act.upper(), rep.upper())
    first_image = cv2.imread(os.path.join(dataset_path,
                                          rel_img_path,
                                          'kin_%s_%s_%s_%s_color_00000.ppm' % (kin, sub, act, rep)),
                             cv2.IMREAD_UNCHANGED)

    ts_corr = os.path.join(dataset_path, 'Kinect', 'Correspondences',
                           'corr_moc_kin%s_%s_%s_%s.txt' % (kin[-2:], sub, act, rep))
    corr = pd.read_csv(ts_corr, sep=' ', header=None)[2]
    ds_length = len(corr)

    # joints reading
    j_file = os.path.join(dataset_path, 'Mocap', 'SkeletalData', data_file + '_pos.csv')
    joints_df = pd.read_csv(j_file)
    R, T, K = read_calib(kin)
    T_C_W = np.eye(4, dtype=float)
    T_C_W[:3, :3] = R
    T_C_W[:3, 3] = T / 1000.  # mm to meter

    # save information in dictionary:
    pred_betas = []
    pred_poses = []
    pred_rotmat = []
    pred_camera = []
    gt_poses = []
    gt_betas = []
    gt_bboxes = []
    gt_kp2d = []
    gt_kp3d = []
    gt_T_CW = []
    gt_K = K
    image_size = list(first_image.shape[:-1])
    image_names = []
    depth_names = []

    for i in tqdm(range(ds_length)):

        image_name = 'kin_%s_%s_%s_%s_color_%05i.ppm' % (kin, sub, act, rep, i)
        depth_name = 'kin_%s_%s_%s_%s_depth_%05i.pgm' % (kin, sub, act, rep, i)

        image = cv2.imread(os.path.join(dataset_path, rel_img_path, image_name), cv2.IMREAD_UNCHANGED)
        # Depth not needed during preprocessing
        # depth = cv2.imread(os.path.join(dataset_path, rel_img_path, depth_name), cv2.IMREAD_UNCHANGED)

        joints_W = get_joints_from_bvh(joints_df.iloc[corr.iloc[i]])
        joints_C = np.einsum('ij,bj->bi', R, joints_W) + T
        joints_C_hom = joints_C / joints_C[:, -1:]
        kp_2d = np.einsum('ij,bj->bi', gt_K, joints_C_hom)

        gt_bbox = [bbox_from_kp2d(kp_2d, image_size)]

        cam = FakeCamera(gt_bbox, K_r=gt_K, img_size=image_size)

        det, norm = cam.preprocess_image(image[np.newaxis])

        rotmat, poses, betas, camera = spin(norm)

        pred_betas.append(betas.squeeze().cpu().detach().numpy())
        pred_poses.append(poses.squeeze().cpu().detach().numpy())
        pred_rotmat.append(rotmat.squeeze().cpu().detach().numpy())
        pred_camera.append(camera.squeeze().cpu().detach().numpy())
        gt_bboxes.append(gt_bbox)
        gt_kp2d.append(kp_2d)
        gt_kp3d.append(joints_W)
        gt_T_CW.append(T_C_W)
        image_names.append(os.path.join(rel_img_path, image_name))
        depth_names.append(os.path.join(rel_img_path, depth_name))

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
    ds_name = 'mhad'
    fname = 'pred_%s_%s_%s_%s_%s.pkl' % (ds_name, kin, sub, act, rep)
    with open(os.path.join(pred_folder, fname), 'wb') as handle:
        pkl.dump(seq_pred, handle)


if __name__ == "__main__":

    data_folder = os.path.join(dataset_path, 'Mocap', 'SkeletalData')
    data_files = [f[:-4] for f in os.listdir(data_folder) if f.endswith('.bvh')]
    for data_file in data_files:
        print(data_file)
        main(data_file)
