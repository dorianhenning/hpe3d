import os


ROOT_3DPW = '/home/dorianhenning/datasets/3DPW/'
ROOT_MHAD = '/home/dorianhenning/datasets/BMHAD'
ROOT_HPE3D = '/home/dorianhenning/git/hpe3d'
DS_PATHS = {'mhad': ROOT_MHAD,
            '3dpw': ROOT_3DPW}
SPIN_MODEL = os.path.join(ROOT_HPE3D, 'data/model_checkpoint.pt')
JOINT_REGRESSOR_TRAIN_EXTRA = os.path.join(ROOT_HPE3D, 'data/J_regressor_extra.npy')
SMPL_MEAN_PARAMS = os.path.join(ROOT_HPE3D, 'data/smpl_mean_params.npz')
SMPL_MODEL_DIR = os.path.join(ROOT_HPE3D, 'data/smpl/')
