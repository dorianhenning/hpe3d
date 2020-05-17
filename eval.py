import numpy as np
import os
import argparse
import cv2
import torch
from tqdm import tqdm
import pickle as pkl

from hpe3d.models import SMPL
import hpe3d.utils.config as cfg

from hpe3d.filter import filter_variable
from hpe3d.utils.img_utils import FakeCamera, project_points
from hpe3d.utils.geometry import convert_smpl_joint_hom, rot6d_to_rotmat
from hpe3d.utils.error import mpjpe
from hpe3d.utils.plotting import plot_error, plot_3d_tracks, plot_xyz_coordinates
from hpe3d.utils.renderer import Renderer
import hpe3d.utils.constants as C

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
NUM_JOINTS = len(C.MHAD_2_HPE3D)


def main(args):

    dataset_path = cfg.DS_PATHS[args.dataset]

    # Vicon Ground Truth, Faster-RCNN, and OKVIS Data
    seq = pkl.load(open(os.path.join(args.pkl_file), 'rb'), encoding='latin1')
    clip_name = os.path.split(args.pkl_file)[1].rstrip('.pkl').lstrip('pred_')

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

    joints_gt_W = np.insert(gt_kp3d.reshape(-1, NUM_JOINTS, 3), 3, 1, axis=2)
    joints_gt_C = np.einsum('bji,bki->bkj', T_C_W, joints_gt_W)

    # Filtering
    betas_filt = filter_variable(pred_betas, mode='c')
    poses_filt = filter_variable(pred_poses, mode='a', f1=100, f2=0.001)

    # correct way:
    pos_filt = filter_variable(T_W_B[:, :3, 3].copy(), mode='a', f1=1000, f2=0.01)

#    # Filtering
#    betas_filt = filter_variable(pred_betas, mode='c')
#    poses_filt = filter_variable(pred_poses, mode='a')
#    pos_filt = filter_variable(T_W_B[:, :3, 3].copy(), mode='a')

    T_W_B_filt = T_W_B.copy()
    T_W_B_filt[:, :3, 3] = pos_filt
    T_C_B_filt = np.matmul(T_C_W, T_W_B_filt)
    T_B_C_filt = np.linalg.inv(T_C_B_filt)

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
    joints_base = smpl_out_base.joints.cpu().numpy()[:, C.SPIN_2_HPE3D]

    joints_baseline_B = convert_smpl_joint_hom(joints_base)
    joints_baseline_C = np.einsum('bji,bki->bkj', T_C_B, joints_baseline_B)
    joints_baseline_W = np.einsum('bji,bki->bkj', T_W_B, joints_baseline_B)

    # SMPL forward pass -- IMPROVED
    rotmat_filt = rot6d_to_rotmat(torch.from_numpy(poses_filt).to(device).float()).view(ds_length, 24, 3, 3)

    smpl_out_impr = smpl(betas=torch.from_numpy(betas_filt).to(device).float(),
                         body_pose=rotmat_filt[:, 1:],
                         global_orient=rotmat_filt[:, 0].unsqueeze(1),
                         pose2rot=False,
                         extra_joints=True)
    joints_impr = smpl_out_impr.joints.cpu().numpy()[:, C.SPIN_2_HPE3D]

    joints_improved_B = convert_smpl_joint_hom(joints_impr)
    joints_improved_C = np.einsum('bji,bki->bkj', T_C_B_filt, joints_improved_B)
    joints_improved_W = np.einsum('bji,bki->bkj', T_W_B_filt, joints_improved_B)

    # Joint Selection
    j_idx = [0, 3, 6, 9, 13, 16]
    j_idx_mhad = [C.MHAD_2_HPE3D[i] for i in j_idx]
#    j_idx_spin = [C.SPIN_2_HPE3D[i] for i in j_idx]
#    j_names = [C.MHAD_JOINT_NAMES[i] for i in j_idx_mhad]

    j_al_baseline_W = joints_baseline_W - np.mean(joints_baseline_W, axis=0)
    j_al_improved_W = joints_improved_W - np.mean(joints_improved_W, axis=0)
    j_al_gt_W = joints_gt_W - np.mean(joints_gt_W, axis=0)

    # Errors all joints:
    err_3d_base = (joints_baseline_C - joints_gt_C)[..., :-1]
    err_3d_impr = (joints_improved_C - joints_gt_C)[..., :-1]

#    err_base_avg = mpjpe(joints_baseline_C[..., :-1], joints_gt_C[..., :-1], align=False)
#    err_impr_avg = mpjpe(joints_improved_C[..., :-1], joints_gt_C[..., :-1], align=False)

    # MPJPE (root joint aligned)
    MPJPE_base = mpjpe(joints_baseline_C[..., :-1], joints_gt_C[..., :-1])
    MPJPE_impr = mpjpe(joints_improved_C[..., :-1], joints_gt_C[..., :-1])

    # Projection of joints on image plane
    proj_joints_base = project_points(joints_baseline_C[:, :, :-1], gt_K)
    proj_joints_impr = project_points(joints_improved_C[:, :, :-1], gt_K)
        
    mse_base = ((j_al_baseline_W - j_al_gt_W) ** 2).mean()
    mse_impr = ((j_al_improved_W - j_al_gt_W) ** 2).mean()

    if args.plot:
        plot_path = os.path.join(dataset_path, 'Plots')

        # MSE Plot
        fname = 'err_mse_%s_pelvis.png' % (clip_name)
        full_path = os.path.join(plot_path, fname)
        plot_error('mse',
                   path=full_path,
                   Pelvis={'Baseline': err_3d_base[:, 0],
                           'Filtered': err_3d_impr[:, 0]})
        fname = 'err_mse_%s_rhand.png' % (clip_name)
        full_path = os.path.join(plot_path, fname)
        plot_error('mse',
                   path=full_path,
                   Right_Hand={'Baseline': err_3d_base[:, 6],
                               'Filtered': err_3d_impr[:, 6]}
                   )

#        # XY & Z Plot
#        fname = 'err_xy_z_%s.png' % (clip_name)
#        full_path = os.path.join(plot_path, fname)
#        plot_error('xyz-error',
#                   pelvis={'base': err_3d_base[:, 0],
#                           'impr': err_3d_impr[:, 0]},
#                   head={'base': err_3d_base[:, 3],
#                         'impr': err_3d_impr[:, 3]}
#                   )
#
#        plot_3d_tracks(pelvis=joints_gt_W[:, 0, :3], prediction=joints_baseline_W[:, 0, :3])

        # Coordinate plots
        for i in [0,3]:#j_idx:
            j_name = C.MHAD_JOINT_NAMES[C.MHAD_2_HPE3D[i]]
            fname = 'coords_%s_%s.png' % (clip_name, j_name)
            full_path = os.path.join(plot_path, fname)
            plot_xyz_coordinates(full_path,
                                 groundtruth=j_al_gt_W[:, i],
                                 baseline=j_al_baseline_W[:, i],
                                 improved=j_al_improved_W[:, i])

    if args.render:

        red_color = (0.3, 0.3, 0.8, 1.0)
        green_color = (0.3, 0.8, 0.3, 1.0)  # cv2: BGR
        blue = (205, 0, 0)
        #blue = (255, 0, 0)
        green = (113, 179, 60)
        #green = (0, 255, 0)
        red = (71, 99, 255)
        #red = (0, 0, 255)

        renderer = Renderer(focal_length=gt_K[(0, 1), (0, 1)],
                            img_res=[640,480],
                            camera_center=gt_K[:2, 2],
                            faces=smpl.faces)
        _, seq_name = os.path.split(args.pkl_file)
#        import pdb;pdb.set_trace()
        #new_folder = os.path.join(dataset_path, 'Render', seq_name[:-4] + '_filt_a_joints')
        #new_folder = os.path.join(dataset_path, 'Render', seq_name[:-4] + '_base_a_joints')
        #new_folder = os.path.join(dataset_path, 'Render', seq_name[:-4] + '_base_a_body')
        new_folder = os.path.join(dataset_path, 'Render', seq_name[:-4] + '_filt_a_body')
        #new_folder = os.path.join(dataset_path, 'Render', seq_name[:-4])
        try:
            os.mkdir(new_folder)
        except FileExistsError:
            pass

        verts_base = smpl_out_base.vertices.cpu().numpy()
        verts_impr = smpl_out_impr.vertices.cpu().numpy()
        for i in tqdm(range(ds_length)):
            if i != 175:
                continue
            img = cv2.imread(image_paths[i])
            root, image_name = os.path.split(image_paths[i])

            verts_base_h = np.insert(verts_base, 3, 1, axis=2)
            verts_base_W = np.einsum('bji,bki->bkj', T_W_B, verts_base_h)[:, :, :-1]
            verts_impr_h = np.insert(verts_impr, 3, 1, axis=2)
            verts_impr_W = np.einsum('bji,bki->bkj', T_W_B, verts_impr_h)[:, :, :-1]

 #           img = renderer(verts_base[i], T_B_C[i].copy(), img.copy(), color=red_color)
            img = renderer(verts_impr[i], T_B_C_filt[i].copy(), img, color=green_color)

   #         for j in range(18):
   #             cv2.circle(img,
   #                        (int(proj_joints_base[i, j, 0]),
   #                         int(proj_joints_base[i, j, 1])), 4, red, -1)
  #              cv2.circle(img,
  #                         (int(proj_joints_impr[i, j, 0]),
  #                          int(proj_joints_impr[i, j, 1])), 4, green, -1)
     #           cv2.circle(img,
     #                      (int(gt_kp2d[i, j, 0]),
     #                       int(gt_kp2d[i, j, 1])), 4, blue, -1)

        cv2.imwrite(os.path.join(new_folder, image_name[:-4] + '.png'), img)
#    return (mse_base, mse_impr)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Evaluate datasets')

    parser.add_argument('--pkl_file', type=str, required=True,
                        help='PKL file from preprocessed dataset')

    parser.add_argument('--dataset', type=str, choices=['mhad', '3dpw'], required=True,
                        help='Name of the dataset to be preprocessed')

    parser.add_argument('--render', type=bool, default=False,
                        help='Should the results be rendered')

    parser.add_argument('--plot', type=bool, default=True,
                        help='Should the results be plotted')

    args = parser.parse_args()

    main(args)
