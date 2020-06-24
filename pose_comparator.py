import numpy as np
from pose_visualizer import PoseVisualizer
from image_aligner import ImageAligner


def distance(k1, k2):
    return np.linalg.norm(np.array(k1[:2])-np.array(k2[:2]))


def compute_distance(poses_1, poses_2):

    assert len(poses_1) == len(poses_2), "List of both poses must be the same length"

    poses_2 = ImageAligner.align_poses(poses_1, poses_2)

    distances = []

    for p in range(len(poses_1)):

        keypoints_1, keypoints_2 = poses_1[p], poses_2[p]

        dist = []

        for k1, k2 in zip(keypoints_1, keypoints_2):

            dist.append(distance(k1, k2))

        distances.append(dist)

    return distances


def find_and_compute_distance(poses_1, poses_2):

    assert len(poses_2) >= len(poses_1), "Second poses list must have equal or larger size than referent (first) poses list"

    all_distances = []

    for i in range(len(poses_2) - len(poses_1)):

        new_poses_2 = poses_2[i:i+len(poses_1)]

        all_distances.append(np.array(compute_distance(poses_1, new_poses_2)).flatten().sum())

    min_idx = 0 if len(all_distances) == 0 else np.where(all_distances == min(all_distances))[0][0]

    min_poses_2 = poses_2[min_idx:min_idx+len(poses_1)]

    return poses_1, min_poses_2, compute_distance(poses_1, min_poses_2)


def demo():
    poses_1, poses_2 = [], [] # Todo load from dataset

    compute_distance(poses_1, poses_2)
    poses_2 = poses_1
    # poses_2 = poses_2[:10] + poses_2 + poses_2[30:]

    find_and_compute_distance(poses_1, poses_2)

    PoseVisualizer.do_plot3D(poses_1)
