from detectron2.data.datasets.builtin_meta import COCO_PERSON_KEYPOINT_NAMES
import matplotlib.pyplot as plt
import numpy as np
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from image_aligner import ImageAligner
import time
import matplotlib as mpl
from detectron2.data.datasets.builtin_meta import COCO_PERSON_KEYPOINT_NAMES
import cv2


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

    @staticmethod
    def break_pose_into_parts(tracked_pose):

        pose_by_parts = dict()

        for pose in tracked_pose:

            for i, part in enumerate(pose):

                if pose_by_parts.get(i) is None:
                    pose_by_parts[i] = list()

                pose_by_parts[i].append(part)

        return pose_by_parts

    def plot3D(self, id, show_all_parts=False):

        tracked_pose = self.poses_for_id(id)

        self.do_plot3D(tracked_pose=tracked_pose,
                       title="Pose for tracker {0}".format(id),
                       show_all_parts=show_all_parts,
                       keypoint_names=self.pose_tracker.visualizer.metadata.keypoint_names)

        """
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
        """


    @staticmethod
    def do_plot3D(tracked_pose, title="Pose", show_all_parts=False, keypoint_names=COCO_PERSON_KEYPOINT_NAMES):

        pose_by_parts = PoseVisualizer.break_pose_into_parts(tracked_pose)

        parts_list = ('nose', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle')

        if show_all_parts:
            parts_list = keypoint_names

        ax = plt.axes(projection="3d")
        legend = []

        for k in pose_by_parts.keys():

            part_name = keypoint_names[k]

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
        plt.title(title)
        plt.show()

        #plt.savefig("./res/3D.png")

    @staticmethod
    def draw_pose(p, title, p_ref=None):
        metadata = MetadataCatalog.get("keypoints_coco_2017_val")
        image = np.zeros((800, 300, 1))
        vis = Visualizer(image, metadata)

        vis.draw_and_connect_keypoints(p)

        if p_ref is not None:

            interest_points = ('left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle')

            for i in range(len(p)):

                if COCO_PERSON_KEYPOINT_NAMES[i] not in interest_points:
                    continue

                k, r = p[i], p_ref[i]
                #vis.draw_line([k[0], r[0]], [k[1], r[1]], color="red")

                vis.output.ax.add_line(
                    mpl.lines.Line2D(
                        [k[0], r[0]],
                        [k[1], r[1]],
                        linewidth=1 * vis.output.scale,
                        color=(0.0, 0.0, 1.0),
                        linestyle="-",
                        marker="."
                    )
                )

        visimg = vis.output.get_image()
        cv2.imshow(title, visimg)
        cv2.waitKey(1)

        return visimg

    @staticmethod
    def show_sequence(poses_1, poses_2, delay=2):

        for i in range(len(poses_1)):

            PoseVisualizer.draw_pose(poses_1[i], "Pose in sequence 1")
            #draw_pose(poses_2[i], "Pose in sequence 2")

            p_al = ImageAligner.align_pose(poses_1[i], poses_2[i])
            PoseVisualizer.draw_pose(p_al, "Pose in sequence 2 aligned", poses_1[i])
            cv2.moveWindow("Pose in sequence 2 aligned", 300, 0)

            time.sleep(delay)

        cv2.destroyAllWindows()
        cv2.waitKey(1)
