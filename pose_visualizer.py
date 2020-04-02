import matplotlib.pyplot as plt
import numpy as np


class PoseVisualizer:

    def __init__(self, pose_tracker):
        #self.keypoint_names = metadata.keypoint_names
        self.pose_tracker = pose_tracker

    def poses_for_id(self, id):

        tracked_poses = self.pose_tracker.tracked_poses
        out = []

        for k in range(min(tracked_poses.keys()), 1+max(tracked_poses.keys())):

            for pose in tracked_poses[k]:
                if pose["id"] == id:
                    out.append(pose["keypoints"].numpy())

        return np.array(out)

    def break_pose_into_parts(self, tracked_pose):

        pose_by_parts = dict()

        for pose in tracked_pose:

            for i, part in enumerate(pose):

                if pose_by_parts.get(i) is None:
                    pose_by_parts[i] = list()

                pose_by_parts[i].append(part)

        return pose_by_parts

    def plot3D(self, id):

        tracked_pose = self.poses_for_id(id)
        pose_by_parts = self.break_pose_into_parts(tracked_pose)

        ax = plt.axes(projection="3d")
        legend = []

        for k in pose_by_parts.keys():

            sc = np.array(pose_by_parts.get(k))
            xs = sc[:, 0]
            zs = sc[:, 1]
            ys = np.arange(0, len(sc))

            ax.plot3D(xs, ys, zs, alpha=0.6)

            legend.append(self.pose_tracker.visualizer.metadata.keypoint_names[k])

        ax.set_xlabel('x')
        ax.set_ylabel('t')
        ax.set_zlabel('y')
        plt.legend(legend)
