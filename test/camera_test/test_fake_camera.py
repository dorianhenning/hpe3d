import torch
import random
import numpy as np
from hpe3d.utils.img_utils import FakeCamera


def test_fake_camera():

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # sanity checks for dimensions and rotation matrix properties
    batch_size = random.randint(1, 1000)
    img_size = [random.randint(100, 1000), random.randint(100, 1000)]
    img_size = torch.randint(100, 1000, (2,), device=device)
    K = torch.eye(3, dtype=torch.float32, device=device)
    K[0, 0] = K[1, 1] = random.randint(50, 300)
    K[0:2, 2] = img_size / 2
    K = K.view(1, 3, 3).repeat(batch_size, 1, 1)

    ul = torch.randint(0, img_size[0], (batch_size, 1), device=device)
    w = torch.randint(1, img_size[0] // 2, (batch_size, 1), device=device)
    ur = torch.clamp(ul + w, 0, img_size[0])
    vt = torch.randint(0, img_size[1], (batch_size, 1), device=device)
    h = torch.randint(1, img_size[1] // 2, (batch_size, 1), device=device)
    vb = torch.clamp(vt + h, 0, img_size[1])
    bboxes = torch.cat((ul, vt, ur, vb), axis=1)
    img_size = img_size.view(1, -1).repeat(batch_size, 1)

    test_cam = FakeCamera(bboxes, K, img_size)

    pred_cam = torch.rand((batch_size, 3), dtype=torch.float32, device=device)
    pred_cam[:, 0] += .5
    pred_cam[:, 1:] *= .2

    trafo = test_cam.compute_body_transformation(pred_cam)
    assert trafo.shape[0] == batch_size
    assert trafo.shape[1] == 4
    assert trafo.shape[2] == 4

    det = np.linalg.det(trafo[:, :3, :3].cpu().numpy())
    np.testing.assert_allclose(det, 1, 1e-6)

    print('test_fake_camera:\tsuccessful!')


test_fake_camera()
