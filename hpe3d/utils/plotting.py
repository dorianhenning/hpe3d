import numpy as np
import trimesh
import pyrender
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


red = mcolors.CSS4_COLORS['red']
red = mcolors.CSS4_COLORS['tomato']
light_red = mcolors.CSS4_COLORS['lightcoral']
blue = mcolors.CSS4_COLORS['mediumblue']
light_blue = mcolors.CSS4_COLORS['cornflowerblue']
green = mcolors.CSS4_COLORS['mediumseagreen']
light_green = mcolors.CSS4_COLORS['limegreen']
yellow = mcolors.CSS4_COLORS['goldenrod']
light_yellow = mcolors.CSS4_COLORS['gold']

gray = mcolors.CSS4_COLORS['dimgray']

c_base = [light_red, light_blue, light_green, light_yellow]
c_impr = [red, blue, green, yellow]
c_gt = [gray, gray, gray, gray]
rgb = ['r', 'g', 'b']

title = ['Pelvis', 'Head', 'Right Wrist', 'Right Foot']


def plot_3d_tracks(path=None, view=(50, 120), **kwargs):
    '''
        plots 3-dimensional trajectories of ground truth,
        baseline predictions, and improved predictions

    '''
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=view[0], azim=view[1])

    for key, arg in kwargs.items():
        if arg.ndim == 2:
            assert arg.shape[1] == 3
            ax.plot(arg[:, 0], arg[:, 1], arg[:, 2], label=key, linewidth=2)
        elif arg.ndim == 3:
            assert arg.shape[1] == 3
            ax.plot(arg[:, 0, -1], arg[:, 1, -1], arg[:, 2, -1], label=key, color='0.75', linewidth=2)
            n_arr = arg.shape[2] - 1
            for i in range(n_arr):
                ax.quiver(arg[::15, 0, -1], arg[::15, 1, -1], arg[::15, 2, -1], arg[::15, 0, i], arg[::15, 1, i], arg[::15, 2, i], length=0.1, color=rgb[i], linewidth=2)

    ax.legend(fontsize=22)
    ax.set_xlabel('\nx [m]', fontsize=22, linespacing=2)
    ax.set_ylabel('\ny [m]', fontsize=22, linespacing=2)
    ax.set_zlabel('\nz [m]', fontsize=22, linespacing=2)
#    set_axes_equal(ax, zoom=1.5)
    ax.set_xlim3d(-1.25, 0.15)
    ax.set_ylim3d(-0.75, 0.65)
    ax.set_zlim3d(0.3, 1.7)

    ax.tick_params(labelsize=20)

    plt.tight_layout()

    if path is None:
        plt.show()
    else:
        plt.savefig(path, facecolor='w', edgecolor='w', dpi=300, quality=100,
                    transparent=False, bbox_inches='tight', pad_inches=0.1)
        plt.close()


def plot_error(plot_type, path=None, **kwargs):
    if plot_type == 'mse':
        f = subplot_mse
    elif plot_type == 'xyz-error':
        f = subplot_xyz_error
    else:
        raise ValueError('Plot type unkown')

    nargs = len(kwargs)
    ncol = min(nargs, 2)
    nrow = 1 if nargs < 3 else nargs // ncol + nargs % ncol

    # make subplots with 2 rows if you have more than 2 kwargs, else make one row
    fig, ax = plt.subplots(nrow, ncol, constrained_layout=True, figsize=(12 * ncol, 8 * nrow))

    keys = [k for k in kwargs.keys()]
    # Coordinate plots
    for i, key in enumerate(keys):
        c = i % 2
        r = i // 2
        val = kwargs[key]
        if nargs == 1:
            f(ax, key, **val)
        elif nargs == 2:
            f(ax[r + c], key, **val)
        else:
            f(ax[r, c], key, **val)

    if nargs % ncol != 0:
        ax[-1, -1].axis('off')

    if path is None:
        plt.show()
    else:
        plt.savefig(path, facecolor='w', edgecolor='w', dpi=300, quality=100,
                    transparent=False, bbox_inches='tight', pad_inches=0.1)
        plt.close()


def subplot_xyz_error(ax, name, **kwargs):
    '''
        plots stacked error for xy and z separate (useful in camera frame)

    '''
    count = 0
    for key, val in kwargs.items():
        assert val.ndim == 2
        assert val.shape[1] == 3
        val_xy = np.linalg.norm(val[:, :-1], axis=1)
        val_xy_mean = np.mean(val_xy)
        val_z_mean = np.mean(val[:, -1])

        ax.plot(val_xy, label='xy %s' % key)
        ax.plot(val[:, -1], label='z %s' % key)

        ax.legend(loc='upper right')
        ax.set_title('Stacked XY-Z Error ' + name)
        ax.text(0.02, (0.95 - count * .07), 'average %s: %.4f (z), %.4f (xy)' % (key, val_z_mean, val_xy_mean),
                horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
        count += 1


def subplot_mse(ax, name, **kwargs):
    '''
        plots stacked square error

    '''
    count = 0
#    c = [(0.8, 0.3, 0.3), (0.3, 0.8, 0.3)]
    c = [red, green]
    for key, val in kwargs.items():
        assert val.ndim == 2
        assert val.shape[1] == 3
        val2 = np.linalg.norm(val, axis=1) #** 2  # error^2 is MSE
        val2_mean = np.mean(val2)

        ax.plot(val2, label=key, color=c[count], linewidth=2.5)

        ax.legend(loc='upper right', fontsize=20)
#        ax.set_title('MSE ' + name)
        ax.text(0.02, (0.95 - count * .07), 'Mean %s: %.4f [m]' % (key, val2_mean),
                horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=20)
        count += 1
    ax.axvline(x=175, color='k', linestyle='--', linewidth=0.5, ymax=0.75)
    ax.margins(0.0)
    ax.set_ylim(0.2, 1)
    ax.set_xlabel('frame', fontsize=22)
    ax.set_ylabel('Absolute Error [m]', fontsize=22)
    ax.tick_params(labelsize=20)



def plot_xyz_coordinates(path=None, **kwargs):
    fig, ax = plt.subplots(3, 1, constrained_layout=True, figsize=(12, 12))

    # Coordinate plots
    for key, val in kwargs.items():
        assert val.ndim == 2
        ax[0].plot(val[:, 0], label=str(key))
        ax[0].set_title('x coordinate')

        ax[1].plot(val[:, 1], label=str(key))
        ax[1].set_title('y coordinate')

        ax[2].plot(val[:, 2], label=str(key))
        ax[2].set_title('z coordinate')

    ax[0].legend()
    ax[1].legend()
    ax[2].legend()

    if path is None:
        plt.show()
    else:
        plt.savefig(path, facecolor='w', edgecolor='w', dpi=300, quality=100,
                    transparent=False, bbox_inches='tight', pad_inches=0.1)
        plt.close()


def show_scene(vertices, faces, camera_pose, image, K, joints=[], color=(0.8, 0.3, 0.3, 1.0)):

        mats = []
        for c in color:

            material = pyrender.MetallicRoughnessMaterial(
                metallicFactor=0.2,
                alphaMode='OPAQUE',
                baseColorFactor=c)
            mats.append(material)

        scene = pyrender.Scene(ambient_light=(0.5, 0.5, 0.5))

        for i, v in enumerate(vertices):
            mesh = trimesh.Trimesh(v, faces)
            rot = trimesh.transformations.rotation_matrix(
                np.radians(180), [1, 0, 0])
            mesh.apply_transform(rot)
            mesh = pyrender.Mesh.from_trimesh(mesh, material=mats[i])

            scene.add(mesh, 'mesh')

        camera = pyrender.IntrinsicsCamera(fx=K[0, 0], fy=K[1, 1],
                                           cx=K[0, 2], cy=K[1, 2])
        scene.add(camera, pose=camera_pose)

        cam = trimesh.creation.axis()
        mesh = pyrender.Mesh.from_trimesh(cam, smooth=False)
        scene.add(mesh, pose=camera_pose)
        scene.add(mesh, pose=np.linalg.inv(camera_pose))
        scene.add(mesh, pose=np.eye(4))

        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1)
        light_pose = np.eye(4)

        light_pose[:3, 3] = np.array([0, -1, 1])
        scene.add(light, pose=light_pose)

        light_pose[:3, 3] = np.array([0, 1, 1])
        scene.add(light, pose=light_pose)

        light_pose[:3, 3] = np.array([1, 1, 2])
        scene.add(light, pose=light_pose)

        for i, j_list in enumerate(joints):
            for j in j_list:
                mesh = trimesh.creation.uv_sphere(0.01, count=[32, 32])
                mesh = pyrender.Mesh.from_trimesh(mesh, material=mats[i])

                pos = np.eye(4)
                pos[:3, 3] = j[:-1]
                scene.add(mesh, 'mesh', pose=pos)

        pyrender.Viewer(scene)


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


def draw_skeleton(ax, joints, c='r'):

    for bone in bone_list:
        x0 = joints[bone[0], 0]
        x1 = joints[bone[1], 0]
        y0 = joints[bone[0], 1]
        y1 = joints[bone[1], 1]
        z0 = joints[bone[0], 2]
        z1 = joints[bone[1], 2]

        ax.plot([x0, x1], [y0, y1], [z0, z1], c)


def draw_camera(ax, trafo=np.eye(4), s=0.05, c='k'):

    cam_points = np.array(cam_symbol, dtype=float) * s
    cam_pos = np.einsum('ji,ki->kj', trafo, cam_points)

    for edge in cam_list:
        x0 = cam_pos[edge[0], 0]
        x1 = cam_pos[edge[1], 0]
        y0 = cam_pos[edge[0], 1]
        y1 = cam_pos[edge[1], 1]
        z0 = cam_pos[edge[0], 2]
        z1 = cam_pos[edge[1], 2]

        ax.plot([x0, x1], [y0, y1], [z0, z1], c)


def draw_axis(ax, trafo=np.eye(4), s=0.05):

    axis_points = np.array(axis_symbol, dtype=float) * s
    pos = np.einsum('ji,ki->kj', trafo, axis_points)

    x = Arrow3D([pos[0, 0], pos[1, 0]], [pos[0, 1], pos[1, 1]], [pos[0, 2], pos[1, 2]], **arrow_prop_dict, color='r')
    y = Arrow3D([pos[0, 0], pos[2, 0]], [pos[0, 1], pos[2, 1]], [pos[0, 2], pos[2, 2]], **arrow_prop_dict, color='g')
    z = Arrow3D([pos[0, 0], pos[3, 0]], [pos[0, 1], pos[3, 1]], [pos[0, 2], pos[3, 2]], **arrow_prop_dict, color='b')

    ax.add_artist(x)
    ax.add_artist(y)
    ax.add_artist(z)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')


def set_axes_radius(ax, origin, radius):
    '''
        From StackOverflow question:
        https://stackoverflow.com/questions/13685386/
    '''
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])


def set_axes_equal(ax, zoom=1.):
    '''
        Make axes of 3D plot have equal scale so that spheres appear as spheres,
        cubes as cubes, etc..  This is one possible solution to Matplotlib's
        ax.set_aspect("equal") and ax.axis("equal") not working for 3D.
        input:
          ax:   a matplotlib axis, e.g., as output from plt.gca().

    '''

    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])

    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0])) / zoom
    set_axes_radius(ax, origin, radius)


def plot_joint_correspondence(j1, j2):
    assert j1.ndim == 2
    assert j2.ndim == 2
    assert j1.shape[0] == j2.shape[0]

    j_cat = np.concatenate((j1[np.newaxis], j2[np.newaxis]), axis=0)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(j1.shape[0]):
        ax.plot(j_cat[:, i, 0], j_cat[:, i, 1], j_cat[:, i, 2])

    plt.show()


axis_symbol = [[0, 0, 0, 1],
               [1, 0, 0, 1],
               [0, 1, 0, 1],
               [0, 0, 1, 1]]

arrow_prop_dict = dict(mutation_scale=20, arrowstyle='->', shrinkA=0, shrinkB=0)

cam_symbol = [[0, 0, 0, 1],  # in homogeneous coordinates
              [-1, -1, 1, 1],
              [1, -1, 1, 1],
              [1, 1, 1, 1],
              [-1, 1, 1, 1]]

cam_list = [[0, 1],
            [0, 2],
            [0, 3],
            [0, 4],
            [1, 2],
            [2, 3],
            [3, 4],
            [4, 1]]

bone_list = [[0, 1],  # l hip
             [0, 2],  # r hip
             [0, 3],  # lower back
             [3, 6],  # middle back
             [6, 9],  # upper back
             [9, 13],
             [9, 14],
             [12, 13],
             [12, 14],
             [12, 15],
             [1, 4],  # left thigh
             [2, 5],  # right thigh
             [4, 7],  # left lower leg
             [5, 8],  # right lower leg
             [7, 10],  # left foot
             [8, 11],  # right foot
             [13, 16],
             [14, 17],
             [16, 18],  # l upper arm
             [17, 19],  # r upper arm
             [18, 20],  # l lower arm
             [19, 21],  # r lower arm
             [20, 22],  # l hand
             [21, 23]]  # r hand
