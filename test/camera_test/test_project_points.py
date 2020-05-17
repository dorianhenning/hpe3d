import numpy as np

from hpe3d.utils.img_utils import project_points


def test_project_points():
    '''
        project points takes B x N x 3 vector
        and returns B x N x 2 projected points
    '''
    batch_size = 100
    n = 30
    focal_length = 300
    image_center = 200

    # check dimensions
    points_3d = np.random.rand(batch_size, n, 3)

    K = np.array([[focal_length, 0., image_center],
                  [0., focal_length, image_center],
                  [0., 0., 1.]], dtype=float)
    points_uv = project_points(points_3d, K)

    assert points_uv.shape[0] == batch_size
    assert points_uv.shape[1] == n
    assert points_uv.shape[2] == 2

    # check calculations
    points_3d = np.array([300., 200., 100.], dtype=float)

    points_uv = project_points(np.reshape(points_3d, (1, 1, -1)), K)
    np.testing.assert_allclose(points_uv.squeeze(), (1100, 800))

    print('test_camera_rotation:\tsuccessful!')


test_project_points()
