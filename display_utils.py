import numpy as np
import os

# 3D viewer
from matplotlib import pyplot as plt
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

edges = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],
    [10,11],[11,12],[0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20]]

def plot_hand_3d(ax, joints_3d, joint_color, s_val):
    ax.scatter3D(joints_3d[:, 0], joints_3d[:, 1], joints_3d[:, 2], joint_color)
    for p in range(joints_3d.shape[0]):
        ax.text(joints_3d[p, 0], joints_3d[p, 1],joints_3d[p, 2], '{0}'.format(p))

    for ie, e in enumerate(edges):
        rgb = colors.hsv_to_rgb([ie/float(len(edges)), s_val, 1.0])
        xx = [joints_3d[e[0], 0], joints_3d[e[1], 0]]
        yy = [joints_3d[e[0], 1], joints_3d[e[1], 1]]
        zz = [joints_3d[e[0], 2], joints_3d[e[1], 2]]
        ax.plot3D(xx, yy, zz, color=rgb)

def plot_hand(joints_2d_screen, joint_color, s_val):
    for p in range(joints_2d_screen.shape[0]):
        plt.plot(joints_2d_screen[p, 0], joints_2d_screen[p, 1], joint_color)
        plt.text(joints_2d_screen[p, 0], joints_2d_screen[p, 1], '{0}'.format(p))
    for ie, e in enumerate(edges):
        rgb = colors.hsv_to_rgb([ie/float(len(edges)), s_val, 1.0])
        plt.plot(joints_2d_screen[e, 0], joints_2d_screen[e, 1], color=rgb)

def display_hand_3d(verts, joints, mano_faces=None, ax=None, alpha=0.2, 
    edge_color=(50 / 255, 50 / 255, 50 / 255)):
    """
    Displays hand batch_idx in batch of hand_info, hand_info as returned by
    generate_random_hand
    """
    if mano_faces is None:
        ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], alpha=0.1)
    else:
        mesh = Poly3DCollection(verts[mano_faces], alpha=alpha)
        face_color = edge_color
        mesh.set_edgecolor(edge_color)
        mesh.set_facecolor(face_color)
        ax.add_collection3d(mesh)
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], color=edge_color)
    cam_equal_aspect_3d(ax, verts)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

def cam_equal_aspect_3d(ax, verts, flip_x=False):
    """
    Centers view on cuboid containing hand and flips y and z axis
    and fixes azimuth
    """
    extents = np.stack([verts.min(0), verts.max(0)], axis=1)
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / 2
    if flip_x:
        ax.set_xlim(centers[0] + r, centers[0] - r)
    else:
        ax.set_xlim(centers[0] - r, centers[0] + r)
    # Invert y and z axis
    ax.set_ylim(centers[1] + r, centers[1] - r)
    ax.set_zlim(centers[2] + r, centers[2] - r)

# 1 use image and joint heat maps as input
# 0 use image only as input
color_hand_joints = [[1.0, 0.0, 0.0],
                     [0.0, 0.4, 0.0], [0.0, 0.6, 0.0], [0.0, 0.8, 0.0], [0.0, 1.0, 0.0],  # thumb
                     [0.0, 0.0, 0.6], [0.0, 0.0, 1.0], [0.2, 0.2, 1.0], [0.4, 0.4, 1.0],  # index
                     [0.0, 0.4, 0.4], [0.0, 0.6, 0.6], [0.0, 0.8, 0.8], [0.0, 1.0, 1.0],  # middle
                     [0.4, 0.4, 0.0], [0.6, 0.6, 0.0], [0.8, 0.8, 0.0], [1.0, 1.0, 0.0],  # ring
                     [0.4, 0.0, 0.4], [0.6, 0.0, 0.6], [0.8, 0.0, 0.8], [1.0, 0.0, 1.0]]  # little

color_hand_joints2 = [[0.0, 1.0, 0.0],
                     [1.0, 0.4, 0.0], [1.0, 0.6, 0.0], [1.0, 0.8, 0.0], [1.0, 1.0, 0.0],  # thumb
                     [0.0, 1.0, 0.6], [0.0, 1.0, 1.0], [0.2, 1.0, 1.0], [0.4, 1.0, 1.0],  # index
                     [1.0, 0.4, 0.4], [1.0, 0.6, 0.6], [1.0, 0.8, 0.8], [1.0, 1.0, 1.0],  # middle
                     [0.4, 0.4, 1.0], [0.6, 0.6, 1.0], [0.8, 0.8, 1.0], [0.0, 1.0, 1.0],  # ring
                     [0.4, 1.0, 0.4], [0.6, 1.0, 0.6], [0.8, 1.0, 0.8], [0.0, 1.0, 1.0]]  # little

def draw_3d_skeleton(pose_cam_xyz, image_size, ax):
    """
    :param pose_cam_xyz: 21 x 3
    :param image_size: H, W
    :return:
    """
    assert pose_cam_xyz.shape[0] == 21

    marker_sz = 15
    line_wd = 2

    for joint_ind in range(pose_cam_xyz.shape[0]):
        ax.plot(pose_cam_xyz[joint_ind:joint_ind + 1, 0], pose_cam_xyz[joint_ind:joint_ind + 1, 1],
                pose_cam_xyz[joint_ind:joint_ind + 1, 2], '.', c=color_hand_joints[joint_ind], markersize=marker_sz)
        if joint_ind == 0:
            continue
        elif joint_ind % 4 == 1:
            ax.plot(pose_cam_xyz[[0, joint_ind], 0], pose_cam_xyz[[0, joint_ind], 1], pose_cam_xyz[[0, joint_ind], 2],
                    color=color_hand_joints[joint_ind], lineWidth=line_wd)
        else:
            ax.plot(pose_cam_xyz[[joint_ind - 1, joint_ind], 0], pose_cam_xyz[[joint_ind - 1, joint_ind], 1],
                    pose_cam_xyz[[joint_ind - 1, joint_ind], 2], color=color_hand_joints[joint_ind],
                    linewidth=line_wd)

    ax.axis('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=-85, azim=-75)