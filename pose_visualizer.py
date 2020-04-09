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

    def plot3D(self, id, show_all_parts=False):

        tracked_pose = self.poses_for_id(id)
        pose_by_parts = self.break_pose_into_parts(tracked_pose)

        parts_list = ('nose', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle')

        if show_all_parts:
            parts_list = self.pose_tracker.visualizer.metadata.keypoint_names

        ax = plt.axes(projection="3d")
        legend = []

        for k in pose_by_parts.keys():

            part_name = self.pose_tracker.visualizer.metadata.keypoint_names[k]

            if part_name not in parts_list:
                continue

            sc = np.array(pose_by_parts.get(k))
            xs = sc[:, 0]
            zs = sc[:, 1]
            ys = np.arange(0, len(sc))

            ax.plot3D(xs, ys, zs, alpha=1)

            legend.append(part_name)

        ax.set_xlabel('x')
        ax.set_ylabel('t')
        ax.set_zlabel('y')
        plt.legend(legend)

        ax.view_init(elev=30, azim=15)
        plt.title("Pose for tracker {0}".format(id))
        plt.show()

        #plt.savefig("./res/3D.png")
