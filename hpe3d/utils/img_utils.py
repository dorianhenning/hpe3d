import torch
import numpy as np
import cv2
from torchvision.transforms import Normalize
import hpe3d.utils.constants as C


class Distortion:

    def __init__(self, K=None, dist_coeffs=None, img_size=(640, 480)):

        if dist_coeffs is None:
            self.dist_coeffs = np.array(C.DISTORTION, dtype=float)
        else:
            self.dist_coeffs = dist_coeffs

        if K is None:
            self.K_R = np.array(C.K_R, dtype=float)
        else:
            self.K_R = K
        R = np.eye(3)
        self.map1, self.map2 = cv2.initUndistortRectifyMap(self.K_R,
                                                           self.dist_coeffs,
                                                           R,
                                                           self.K_R,
                                                           img_size,
                                                           cv2.CV_32FC1)

    def undistort(self, img):
        new_img = cv2.remap(img, self.map1, self.map2, cv2.INTER_LINEAR)

        return new_img


def project_points(points, K=np.array(C.K_R, dtype=float)):
    '''
    Batch size B
    Points per batch N
    points: B x N x 3
    '''

    point_uv = np.einsum('ji,bni->bnj', K, points)
    point_c = point_uv / (point_uv[:, :, 2:] + 1e-8)

    return point_c[..., :-1]


class FakeCamera:

    def __init__(
            self,
            bbox,
            K_r=C.K_R,
            img_size=C.IMG_SIZE,
            enlarge=None,
            preprocess=True,
    ):

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.batch_size = len(bbox)

        # bbox = [u_l, v_t, u_r, v_b]
        self.bbox = torch.tensor(bbox, dtype=torch.int64, device=self.device)

        self.K_r = torch.tensor(K_r, dtype=torch.float32, device=self.device)
        if self.K_r.ndim == 2:
            self.K_r = self.K_r.view(1, 3, 3).repeat(self.batch_size, 1, 1)

        self.K_r_inv = torch.inverse(self.K_r)
        self.focal_length = self.K_r[:, 0, 0]

        self.img_size = torch.tensor(img_size, dtype=torch.int64, device=self.device)
        if self.img_size.ndim == 1:
            self.img_size = self.img_size.view(1, 2).repeat(self.batch_size, 1)

        # fake camera
        self._compute_camera_rotation()
        self._compute_fake_camera()
        self.K_f_inv = torch.inverse(self.K_f)

        self.preprocess = preprocess
        if self.preprocess:
            self.normalize_img = Normalize(mean=C.IMG_NORM_MEAN, std=C.IMG_NORM_STD)
            self.map_x, self.map_y = self._compute_remapping()

    def _compute_camera_rotation(self):
        u_l, v_t, u_r, v_b = self.bbox[:, 0], self.bbox[:, 1], self.bbox[:, 2], self.bbox[:, 3]

        w_bb = u_r - u_l
        h_bb = v_b - v_t

        u_m = u_l + w_bb / 2.0
        v_m = v_t + h_bb / 2.0

        ones = torch.ones(self.batch_size, dtype=torch.float32, device=self.device)
        zeros = torch.zeros(self.batch_size, dtype=torch.float32, device=self.device)

        x_m = torch.cat((u_m, v_m, ones)).reshape(3, self.batch_size)
        x_m_r = torch.einsum('bij,jb->ib', self.K_r_inv, x_m)
        x_f_r = torch.tensor([[0., 0., 1.]], dtype=torch.float32, device=self.device).T.repeat(1, self.batch_size)

        # normalize vectors
        x_m_r = x_m_r / torch.norm(x_m_r, dim=0, keepdim=True)

        # np.dot(x_fp_r, x_m_r) == x_m_r[-1] since all other entries in x_fp_r are 0.0
        self.theta = torch.acos(x_m_r[-1])

        # Rodrigues Formula
        k = torch.cross(-x_m_r, x_f_r)  # x_m_r is a horizontal stack of vectors

        # result is a vertical stack of vectors
        k = k / torch.norm(k, dim=0, keepdim=True)

        K = torch.empty((self.batch_size, 3, 3), dtype=torch.float32, device=self.device)
        K[:, 0, 0] = zeros
        K[:, 0, 1] = -k[2]
        K[:, 0, 2] = k[1]
        K[:, 1, 0] = k[2]
        K[:, 1, 1] = zeros
        K[:, 1, 2] = -k[0]
        K[:, 2, 0] = -k[1]
        K[:, 2, 1] = k[0]
        K[:, 2, 2] = zeros

        R = torch.eye(3, dtype=torch.float32, device=self.device) \
            + torch.sin(self.theta).view(-1, 1, 1) * K \
            + (1 - torch.cos(self.theta)).view(-1, 1, 1) * torch.bmm(K, K)

        self.R_BC = R
        self.R_CB = torch.inverse(R)

    def _compute_fake_camera(self):

        self.K_f = torch.zeros_like(self.K_r)
        self.K_f[:, :2, 2] = C.IMG_RES / 2.
        self.K_f[:, 2, 2] = 1.

        M = torch.bmm(self.R_CB, self.K_r_inv)  # R_CB = R_inv

        # corner candidates of bounding box
        corners = torch.ones((self.batch_size, 4, 3), dtype=torch.float32, device=self.device)
        corners[:, 0, 0] = self.bbox[:, 0]  # u_l
        corners[:, 1, 0] = self.bbox[:, 0]  # u_l
        corners[:, 2, 0] = self.bbox[:, 2]  # u_r
        corners[:, 3, 0] = self.bbox[:, 2]  # u_r
        corners[:, 0, 1] = self.bbox[:, 1]  # v_t
        corners[:, 1, 1] = self.bbox[:, 3]  # v_t
        corners[:, 2, 1] = self.bbox[:, 1]  # v_b
        corners[:, 3, 1] = self.bbox[:, 3]  # v_b

        support = torch.einsum('bij,bkj->bki', M, corners)
        support *= 2. / C.IMG_RES
        support.pow_(-1)  # reciprocal
        support = torch.abs(support[..., :2]).view(self.batch_size, -1)

        f, _ = torch.min(support, dim=1)

        self.K_f[:, 0, 0] = f  # f_x == f_y (square image patch)
        self.K_f[:, 1, 1] = f

    def _compute_remapping(self):

        M = torch.bmm(self.K_r, torch.bmm(self.R_BC, self.K_f_inv))  # R_BC = R
        X_tmp = torch.empty((self.batch_size, C.IMG_RES, C.IMG_RES, 3), dtype=torch.float32, device=self.device)
        u = torch.linspace(0, C.IMG_RES - 1, C.IMG_RES, dtype=torch.float32, device=self.device)
        v = torch.linspace(0, C.IMG_RES - 1, C.IMG_RES, dtype=torch.float32, device=self.device)
        grid_u, grid_v = torch.meshgrid(u, v)
        X_tmp[..., 0] = grid_u
        X_tmp[..., 1] = grid_v
        X_tmp[..., 2] = 1.

        X_tmp = torch.einsum('bij,bklj->bkli', M, X_tmp)
        X_tmp = X_tmp / X_tmp[..., 2:]
        X_tmp = X_tmp[..., :2].long()

        uv_min = torch.zeros((self.batch_size, C.IMG_RES, C.IMG_RES, 2), dtype=torch.int64, device=self.device)
        uv_max = self.img_size.view(self.batch_size, 1, 1, 2).repeat(1, C.IMG_RES, C.IMG_RES, 1) - 1

        map_x = torch.min(uv_max[..., 0], torch.max(uv_min[..., 0], X_tmp[..., 0]))
        map_y = torch.min(uv_max[..., 1], torch.max(uv_min[..., 1], X_tmp[..., 1]))

        map_x = torch.transpose(map_x, 1, 2)
        map_y = torch.transpose(map_y, 1, 2)

        return map_x, map_y

    def preprocess_image(self, img, remap=True):
        if not self.preprocess:
            raise Exception('The class was not initialized for preprocessing')

        if not isinstance(img, torch.Tensor):
            img = torch.tensor(img, dtype=torch.uint8, device=self.device)

        mask = torch.zeros(img.shape, dtype=torch.uint8, device=self.device)

        for i in range(self.batch_size):
            mask[i, self.bbox[i, 1]:self.bbox[i, 3], self.bbox[i, 0]:self.bbox[i, 2], :] = 1

        masked_img = img * mask
        det_crop = torch.empty((self.batch_size, C.IMG_RES, C.IMG_RES, 3), device=self.device)
        if remap:
            for i in range(self.batch_size):
                det_crop[i] = masked_img[i, self.map_y[i], self.map_x[i]]
        else:
            raise NotImplementedError('This needs to be implemented for batches')

        det_crop = det_crop.type(torch.float32) / 255.
        det_crop = det_crop.permute(0, 3, 1, 2)
        norm_crop = torch.empty_like(det_crop)
        for i in range(self.batch_size):
            norm_crop[i] = self.normalize_img(det_crop[i].clone())[None]

        return det_crop, norm_crop

    def compute_body_transformation(self, cam_t):
        # compute larger side of the bounding box for scaling coefficient
        bb_x = self.bbox[:, 2] - self.bbox[:, 0]
        bb_y = self.bbox[:, 3] - self.bbox[:, 1]
        bb_scale, _ = torch.max(torch.cat((bb_x, bb_y)).view(-1, self.batch_size), dim=0)

        # get correct predicted distance from camera
        # build Transformation matrix with R & T

        if not isinstance(cam_t, torch.Tensor):
            cam_t = torch.tensor(cam_t, dtype=torch.float32, device=self.device)

        assert cam_t.ndim == 2

        flip_z = torch.tensor([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=torch.float32, device=self.device)
        cam_t = torch.einsum('ij,bj->bi', flip_z, cam_t)
        cam_t[:, 2].pow_(-1)
        cam_t[:, 2] *= 2 * self.focal_length
        cam_t[:, 2].div_(bb_scale)
        cam_t.div_(torch.cos(self.theta).view(-1, 1))

        cam_t = torch.cat((self.R_BC, cam_t.unsqueeze(2)), dim=2)

        last_row = torch.tensor([[[0., 0., 0., 1.]]], dtype=torch.float32, device=self.device).repeat(self.batch_size, 1, 1)
        cam_t = torch.cat((cam_t, last_row), dim=1)

        cam_t[:, 0, 3] *= -1.
        cam_t[:, 1, 0] *= -1.
        cam_t[:, 0, 1] *= -1.
        cam_t[:, 1, 2] *= -1.
        cam_t[:, 2, 1] *= -1.

        Rx = torch.tensor([[1, 0, 0, 0],
                           [0, -1, 0, 0],
                           [0, 0, -1, 0],
                           [0, 0, 0, 1]], dtype=torch.float32, device=self.device)

        cam_t = torch.einsum('bij,jk->bik', cam_t, Rx)

        return cam_t
