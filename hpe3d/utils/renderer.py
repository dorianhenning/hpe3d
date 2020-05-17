import torch
from torchvision.utils import make_grid
import numpy as np
import pyrender
import trimesh
import hpe3d.utils.constants as C


class Renderer:
    '''
        Renderer used for visualizing the SMPL model
        Code adapted from https://github.com/vchoutas/smplify-x
    '''
    def __init__(
        self,
        focal_length=C.FOCAL_LENGTH,
        img_res=C.IMG_SIZE,
        camera_center=C.CAMERA_CENTER,
        faces=None
    ):
        self.renderer = pyrender.OffscreenRenderer(viewport_width=img_res[0],
                                                   viewport_height=img_res[1],
                                                   point_size=1.0)
        self.focal_length = focal_length
        self.camera_center = camera_center
        self.faces = faces

    def visualize_tb(self, vertices, camera_translation, images):
        vertices = vertices.cpu().numpy()
        camera_translation = camera_translation.cpu().numpy()
        images = images.cpu()
        images_np = np.transpose(images.numpy(), (0, 2, 3, 1))
        rend_imgs = []
        for i in range(vertices.shape[0]):
            rend_img = torch.from_numpy(
                np.transpose(
                    self.__call__(vertices[i],
                                  camera_translation[i],
                                  images_np[i]),
                    (2, 0, 1))).float()

            rend_imgs.append(images[i])
            rend_imgs.append(rend_img)
        rend_imgs = make_grid(rend_imgs, nrow=2)
        return rend_imgs

    def __call__(self, vertices, camera_pose, image, color=(0.8, 0.3, 0.3, 1.0)):
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.2,
            alphaMode='OPAQUE',
            baseColorFactor=color)

        mesh = trimesh.Trimesh(vertices, self.faces)

        # Rotate mesh 180 deg around x (pyrender coordinate frame)
        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)
        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        # Rotate trafo 180 deg around x (pyrender coordinate frame)
        Rx = np.array([[1, 0, 0, 0],
                       [0, -1, 0, 0],
                       [0, 0, -1, 0],
                       [0, 0, 0, 1]], dtype=float)
        camera_pose = np.dot(camera_pose, Rx)

        scene = pyrender.Scene(ambient_light=(0.5, 0.5, 0.5))
        scene.add(mesh, 'mesh')

        camera = pyrender.IntrinsicsCamera(fx=self.focal_length[0], fy=self.focal_length[1],
                                           cx=self.camera_center[0], cy=self.camera_center[1])
        scene.add(camera, pose=camera_pose)

        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1)
        light_pose = np.eye(4)

        light_pose[:3, 3] = np.array([0, -1, 1])
        scene.add(light, pose=light_pose)

        light_pose[:3, 3] = np.array([0, 1, 1])
        scene.add(light, pose=light_pose)

        light_pose[:3, 3] = np.array([1, 1, 2])
        scene.add(light, pose=light_pose)

        color, rend_depth = self.renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        valid_mask = (rend_depth > 0)[:, :, None]
        output_img = (color[:, :, :3] * valid_mask +
                      (1 - valid_mask) * image).astype(np.uint8)
        return output_img
