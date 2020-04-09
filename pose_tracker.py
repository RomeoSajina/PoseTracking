import multiprocessing as mp
import cv2
import numpy as np
import pickle
import os

from detectron2.config import get_cfg

from track.predictor import Visualization

from nanonets_object_tracking.deepsort import *

from image_aligner import ImageAligner
from pose_visualizer import PoseVisualizer

import sys
sys.path.insert(0, './nanonets_object_tracking') # Fix torch model loading

# constants
WINDOW_NAME = "PoseTracking"
#CROPPED_WINDOW = "Cropped_{0}"


class PoseTracker:

    def __init__(self, video="./video/out.avi"):

        self.video = video
        self.config_file = "./detectron2/configs/COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml"
        self.opts = ["MODEL.WEIGHTS", "detectron2://COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x/138363331/model_final_997cc7.pkl"]

        self.detection_config_file = "./detectron2/configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
        self.detection_opts = ["MODEL.WEIGHTS", "detectron2://COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl"]

        self.tracking_model = "./nanonets_object_tracking/ckpts/model640.pt"

        self.deepsort = deepsort_rbc(self.tracking_model)

        self.aligner = ImageAligner()

        self.pose_visualizer = PoseVisualizer(self)

        self.dets = dict()
        self.tracked_poses = dict()
        self.cfg = None
        self.visualizer = None
        self.frame_id = 1

        self.crop_padding = 10 # Padding to add when cropping the detection from image

        self.init()

    @staticmethod
    def _find_tracker_by_id(_dets, _id):

        for _tracked in _dets["tracked"]:

            if _tracked["tracker"][0] == _id:
                return _tracked

        return None

    def _find_pose_within_tracker(self, _dets, _tracker):

        assert len(_tracker["overlap_detections"]) > 0, "Overlap detection not found"
        assert len(_tracker["overlap_detections"]) < 2, "To many overlap detection found"

        _tracker_bbox = _tracker["overlap_detections"][0][1:5]
        _instances = _dets["predictions"]["instances"]

        for _i, _bbox in enumerate(self.convert_to_tlwh(_instances.get("pred_boxes").tensor.numpy())):

            if _tracker_bbox == [int(x) for x in _bbox]:
                return _instances.get("pred_keypoints")[_i]

        return None

    @staticmethod
    def _find_best_overlaps(_tracker, _overlap_detections):
        # https://math.stackexchange.com/questions/2449221/calculating-percentage-of-overlap-between-two-rectangles
        _scores = []
        l0, t0, r0, b0 = _tracker[1], _tracker[2], _tracker[1] + _tracker[3], _tracker[2] + _tracker[4]

        for _det in _overlap_detections:
            _d = _det.tlwh
            l1, t1, r1, b1 = _d[0], _d[1], _d[0] + _d[2], _d[1] + _d[3]

            overlap = (max(l0, l1) - min(r0, r1)) * (max(t0, t1) - min(b0, b1))

            _scores.append(overlap)

        return [_overlap_detections[np.argmax(_scores)]]

    def init(self):

        self.dets = dict()
        self.tracked_poses = dict()

        mp.set_start_method("spawn", force=True)

        self.cfg = self.setup_cfg(self.config_file, self.opts)
        self.visualizer = Visualization(self.cfg)

        self.detection_cfg = self.setup_cfg(self.detection_config_file, self.detection_opts)
        self.detection_visualizer = Visualization(self.detection_cfg)

        self.frame_id = 1

    def save(self):
        out = self.dets, self.tracked_poses, self.frame_id

        if not os.path.isdir(".cache/"):
            os.mkdir(".cache/")

        with open(".cache/{0}.pickle".format(self.video.split("/")[-1].split(".")[0]), "wb") as f:
            pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self):
        with open(".cache/{0}.pickle".format(self.video.split("/")[-1].split(".")[0]), "rb") as f:
            self.dets, self.tracked_poses, self.frame_id = pickle.load(f)

    def setup_cfg(self, config_file, opts):

        confidence_threshold = 0.5

        # Pose estimation
        #config_file = self.config_file
        #opts = self.opts

        # load config from file and command-line arguments
        cfg = get_cfg()
        cfg.merge_from_file(config_file)
        cfg.merge_from_list(opts)

        # Set score_threshold for builtin models
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = confidence_threshold
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
        cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = confidence_threshold

        cfg.MODEL.DEVICE = "cpu"
        cfg.freeze()

        return cfg

    def convert_to_tlwh(self, arr):
        """out: (top left x, top left y, width, height)"""
        out = []

        for a in arr:

            x0, y0, x1, y1 = a
            width = x1 - x0
            height = y1 - y0

            out.append([x0, y0, width, height])

        return np.array(out)

    def add_boxes(self, image, frame_id=-1):

        if frame_id == -1:
            frame_id = len(self.dets)

        info = self.dets[frame_id]

        for tracked in info["tracked"]:
            track, overlap_detections = tracked["tracker"], tracked["overlap_detections"]

            track_id, x0, y0, width, height, confidence = track

            #Draw bbox from tracker.
            cv2.rectangle(image, (x0, y0), (x0+width, y0+height), (255, 255, 255), 2)
            cv2.putText(image, str(track_id), (x0, y0), 0, 5e-3 * 200, (0, 255, 0), 2)

            #Draw bbox from detector. Just to compare.
            for det in overlap_detections:
                track_id, x0, y0, width, height, confidence = det
                cv2.rectangle(image, (x0, y0), (x0+width, y0+height), (255, 255, 0), 2)

        image = self.visualizer.draw_only_keypoints(image, info["predictions"]["instances"])

        return image

    def show(self, image, add_boxes=True):

        if add_boxes:
            image = self.add_boxes(image)

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.imshow(WINDOW_NAME, image)

        return cv2.waitKey(1) == 27 # Esc

    def detect(self, frame):
        """
        predictions, visualized_output = self.detection_visualizer.run_on_image(frame)

        person_indicies = np.where(predictions["instances"].get("pred_classes") == np.where(np.array(self.detection_visualizer.metadata.thing_classes) == "person")[0][0])[0]

        predictions["instances"].get_fields()["pred_boxes"] = predictions["instances"].get("pred_boxes")[person_indicies]
        predictions["instances"].get_fields()["scores"] = predictions["instances"].get("scores")[person_indicies]
        predictions["instances"].get_fields()["pred_classes"] = predictions["instances"].get("pred_classes")[person_indicies]
        """

        predictions, visualized_output = self.visualizer.run_on_image(frame)

        cv2.imshow("visualized_output", visualized_output.get_image()[:, :, ::-1])
        cv2.waitKey(1)

        #plt.imshow(visualized_output.get_image()[:, :, ::-1])
        #plt.savefig("./res/detection.png")

        detections = predictions["instances"].get("pred_boxes").tensor.numpy()
        out_scores = predictions["instances"].get("scores").numpy()
        detections = self.convert_to_tlwh(detections)

        return predictions, detections, out_scores

    def track(self, frame, detections, out_scores):

        tracked = []

        if detections is None:
            print("No dets")
            return None

        tracker, overlap_detections = self.deepsort.run_deep_sort(frame, out_scores, detections)

        #track = tracker.tracks[0]
        for track in tracker.tracks:
            info = {"tracker": [], "overlap_detections": []}

            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            bbox = track.to_tlwh()
            info["tracker"] = [track.track_id, int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), None]

            #for det in overlap_detections:
            for det in self._find_best_overlaps(info["tracker"], overlap_detections):
                bbox = det.tlwh
                info["overlap_detections"].append([-1, int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), det.confidence])

            tracked.append(info)

        return np.array(tracked)

    def run(self, track_pose=True):

        cap = cv2.VideoCapture(self.video)
        self.init()

        prev_frame = None

        while cap.isOpened(): # and self.frame_id < 10:

            success, frame = cap.read()

            if not success:
                break

            predictions, detections, out_scores = self.detect(frame)

            tracked = self.track(frame, detections, out_scores)

            self.dets[self.frame_id] = {"predictions": predictions, "detections": detections, "out_scores": out_scores, "tracked": tracked}

            if track_pose and prev_frame is not None:
                self.track_pose(prev_frame, frame)

            stop = self.show(frame)

            if stop:
                break

            self.frame_id += 1

            prev_frame = frame

        cv2.destroyAllWindows()
        cv2.waitKey(1)

    def do_pose_tracking(self, tracker_id=None):

        cap = cv2.VideoCapture(self.video)
        self.init()
        self.load()

        old_frame_id = self.frame_id

        self.frame_id = 1

        prev_frame = None

        while cap.isOpened() and self.frame_id < old_frame_id:

            success, frame = cap.read()

            if not success:
                break

            if prev_frame is not None:
                self.track_pose(prev_frame, frame, tracker_id)

            self.frame_id += 1

            prev_frame = frame

        cv2.destroyAllWindows()
        cv2.waitKey(1)

    def track_pose(self, prev_image, curr_image, tracker_id=None):
        """
        prev_image, curr_image, tracker_id = prev_frame, frame, tracker_id
        """
        prev_dets = self.dets[self.frame_id - 1]
        curr_dets = self.dets[self.frame_id]

        self.tracked_poses[self.frame_id] = list()

        #tracked = prev_dets["tracked"][0]
        for tracked in prev_dets["tracked"]:

            p_id, p_x0, p_y0, p_w, p_h, p_conf = tracked["tracker"]

            if tracker_id is not None and tracker_id != p_id:
                print("Skipping tracker " + str(p_id))
                continue

            p_x0, p_y0, p_w, p_h = self.add_padding((p_x0, p_y0, p_w, p_h))

            prev_bbox = prev_image[p_y0:p_y0+p_h, p_x0:p_x0+p_w]

            target = self._find_tracker_by_id(curr_dets, p_id)

            if target is None:
                print("Target not found " + str(p_id))
                # TODO: handle this
                continue

            c_id, c_x0, c_y0, c_w, c_h, c_conf = target["tracker"]
            c_x0, c_y0, c_w, c_h = self.add_padding((c_x0, c_y0, c_w, c_h))
            curr_bbox = curr_image[c_y0:c_y0+c_h, c_x0:c_x0+c_w]

            #t_curr_bbox = self.aligner.perform_homography(prev_bbox, curr_bbox)
            t_curr_bbox = curr_bbox

            t_curr_bbox = self.aligner.resize(t_curr_bbox)

            #plt.imshow(t_curr_bbox[:, :, ::-1])
            #plt.savefig("./res/resized.png")

            #t_predictions, t_detections, t_out_scores = self.detect(t_curr_bbox)

            tmp_img = np.zeros_like(curr_image)
            tmp_img[0:t_curr_bbox.shape[0]][:, 0:t_curr_bbox.shape[1]] = t_curr_bbox
            t_predictions, t_detections, t_out_scores = self.detect(tmp_img)

            if len(t_predictions["instances"]) == 0:
                print("pose not found" + str(c_id))
                continue

            # tmp fix take the one with most confidence (not really good approach)
            indicies = np.array([np.argmax(t_predictions["instances"].get("scores")).numpy()])
            t_predictions["instances"].get_fields()["scores"] = t_predictions["instances"].get("scores")[indicies]
            t_predictions["instances"].get_fields()["pred_boxes"] = t_predictions["instances"].get("pred_boxes")[indicies]
            t_predictions["instances"].get_fields()["pred_keypoints"] = t_predictions["instances"].get("pred_keypoints")[indicies]
            # /tmp fix

            assert len(t_predictions["instances"].get("pred_keypoints")) == 1, "Error: expected one keypoints prediction"

            pred_keypoints = t_predictions["instances"].get("pred_keypoints")[0]

            """
            cv2.imshow("curr", curr_bbox)
            cv2.imshow("prev", prev_bbox)
            cv2.imshow("t_curr_bbox", t_curr_bbox)
            cv2.imshow("tmp_img", tmp_img)
            cv2.waitKey(1)
            """
            #pred_keypoints = self._find_pose_within_tracker(curr_dets, target)
            #assert pred_keypoints is not None, "Pred_keypoints not found for tracker {0}".format(c_id)

            self.tracked_poses[self.frame_id].append({"id": p_id, "keypoints": pred_keypoints})

    def add_padding(self, bbox):
        x0, y0, w, h = bbox

        def nvl(val):
            return 0 if val < 0 else val

        return nvl(x0 - self.crop_padding), nvl(y0 - self.crop_padding), w + self.crop_padding * 2, h + self.crop_padding * 2




"""
pt = PoseTracker()
pt.run(track_pose=False)
pt.do_pose_tracking(tracker_id=1)
pt.pose_visualizer.plot3D(1)

pt.run()
pt.save()
pt.load()
self = pt

self.pose_visualizer.plot3D(1)
self.pose_visualizer.plot3D(4)
self.pose_visualizer.plot3D(7)

pt2 = PoseTracker(video="./video/DSC_2354.MOV")
pt2.run(track_pose=False)
pt2.save()

pt2.do_pose_tracking(tracker_id=7)
pt2.pose_visualizer.plot3D(7)

pt2.run()
pt2.load()
pt2.save()
self = pt2
"""

