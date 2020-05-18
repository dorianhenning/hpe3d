import numpy as np
import os
import argparse
import cv2
import torch
import trimesh
from tqdm import tqdm
import pickle as pkl

from hpe3d.models import SMPL
import hpe3d.utils.config as cfg

from hpe3d.filter import filter_variable
from hpe3d.utils.img_utils import FakeCamera
from hpe3d.utils.geometry import rot6d_to_rotmat
from hpe3d.utils.display_scenes import display_scenes
import hpe3d.utils.constants as C

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
NUM_JOINTS = len(C.MHAD_2_HPE3D)


def main(args):

    dataset_path = cfg.DS_PATHS[args.dataset]

    # Vicon Ground Truth, Faster-RCNN, and OKVIS Data
    seq = pkl.load(open(os.path.join(args.pkl_file), 'rb'), encoding='latin1')
#    clip_name = os.path.split(args.pkl_file)[1].rstrip('.pkl').lstrip('pred_')

    # clip clip
    ds_length = len(seq['image_names'])

    # read seq file
    pred_betas = np.array(seq['pred_betas'])
    pred_poses = np.array(seq['pred_poses'])
    pred_rotmats = np.array(seq['pred_rotmats'])
    pred_cameras = np.array(seq['pred_cameras'])
    if seq['gt_poses']:
        gt_poses = np.array(seq['gt_poses'])
    if seq['gt_betas']:
        gt_betas = np.array(seq['gt_betas'])
    gt_kp3d = np.array(seq['gt_kp3d'])
    gt_kp2d = np.array(seq['gt_kp2d'])
    gt_bboxes = np.array(seq['gt_bboxes'])
    gt_T_CW = np.array(seq['gt_T_CW'])
    gt_K = np.array(seq['gt_K'])
    image_size = np.array(seq['image_size'])
    image_paths = [os.path.join(dataset_path, i_name) for i_name in seq['image_names']]
    if seq['depth_names']:
        depth_paths = [os.path.join(dataset_path, d_name) for d_name in seq['depth_names']]

    # Get joint subset:
    gt_kp3d = gt_kp3d[:, C.MHAD_2_HPE3D] / 1000.
    gt_kp2d = gt_kp2d[:, C.MHAD_2_HPE3D]

    # Camera Preprocessing:
    cams = FakeCamera(gt_bboxes, gt_K, image_size, preprocess=False)

    T_B_C = cams.compute_body_transformation(pred_cameras).cpu().numpy()
    T_C_B = np.linalg.inv(T_B_C)
    T_C_W = gt_T_CW
    T_W_C = np.linalg.inv(T_C_W)
    T_W_B = np.matmul(T_W_C, T_C_B)

    # Filtering
    betas_filt = filter_variable(pred_betas, mode='c')
    poses_filt = filter_variable(pred_poses, mode='v', f1=100, f2=0.01)

    pos_filt = filter_variable(T_W_B[:, :3, 3].copy(), mode='v', f1=100, f2=0.01)

    T_W_B_filt = T_W_B.copy()
    T_W_B_filt[:, :3, 3] = pos_filt
    T_C_B_filt = np.matmul(T_C_W, T_W_B_filt)

    # SMPL batch NN:
    smpl = SMPL(cfg.SMPL_MODEL_DIR,
                batch_size=ds_length,
                create_transl=False).to(device)

    # SMPL forward pass -- BASELINE
    smpl_out_base = smpl(betas=torch.from_numpy(pred_betas).to(device).float(),
                         body_pose=torch.from_numpy(pred_rotmats[:, 1:]).to(device).float(),
                         global_orient=torch.from_numpy(pred_rotmats[:, 0]).to(device).float().unsqueeze(1),
                         pose2rot=False,
                         extra_joints=True)

    # SMPL forward pass -- IMPROVED
    rotmat_filt = rot6d_to_rotmat(torch.from_numpy(poses_filt).to(device).float()).view(ds_length, 24, 3, 3)

    smpl_out_impr = smpl(betas=torch.from_numpy(betas_filt).to(device).float(),
                         body_pose=rotmat_filt[:, 1:],
                         global_orient=rotmat_filt[:, 0].unsqueeze(1),
                         pose2rot=False,
                         extra_joints=True)

    # get SMPL vertices
    verts_base = smpl_out_base.vertices.cpu().numpy()
    verts_impr = smpl_out_impr.vertices.cpu().numpy()

    for i in tqdm(range(ds_length)):
        img = cv2.imread(os.path.join(image_paths[i]))[..., ::-1]

        scene = trimesh.Scene()

        cam = trimesh.scene.Camera(fov=(50, 70))
        geom = trimesh.creation.camera_marker(cam, marker_height=0.5, origin_size=0.05)
        scene.add_geometry(geom[0], transform=T_W_C[i], node_name='livecam')
        scene.add_geometry(geom[0], transform=T_W_C[i], node_name='livecam_frame')

        # BASELINE
        # Rotate mesh 180 deg around x (coordinate frame changes)
        rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        human_1 = trimesh.Trimesh(verts_base[i], smpl.faces)
        human_1.apply_transform(rot)
        scene.add_geometry(human_1, transform=T_W_B[i], node_name='human_1')

        # IMPROVED
        # Rotate mesh 180 deg around x (coordinate frame changes)
        rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        human_2 = trimesh.Trimesh(verts_impr[i], smpl.faces)
        human_2.apply_transform(rot)
        scene.add_geometry(human_2, transform=T_W_B_filt[i], node_name='human_2')

        # Origin
        origin = trimesh.creation.axis(transform=np.eye(4), origin_size=0.05)
        scene.add_geometry(origin, transform=np.eye(4), node_name='origin')

        # Body Frame
        origin = trimesh.creation.axis(transform=np.eye(4), origin_size=0.05)
        scene.add_geometry(origin, transform=T_W_B[i], node_name='T_W_B_base')

        # Body Frame filtered
        origin = trimesh.creation.axis(transform=np.eye(4), origin_size=0.05)
        scene.add_geometry(origin, transform=T_W_B_filt[i], node_name='T_W_B_impr')

        yield {'rgb': img, 'scene': scene}


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Show demo scene of dataset')

    parser.add_argument('--pkl_file', type=str, required=True,
                        help='PKL file from preprocessed dataset')

    parser.add_argument('--dataset', type=str, choices=['mhad', '3dpw'], required=True,
                        help='Name of the dataset to be preprocessed')

    args = parser.parse_args()

    scenes = main(args)
    display_scenes(scenes)
