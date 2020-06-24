# https://www.dlology.com/blog/how-to-train-detectron2-with-custom-coco-datasets/

from detectron2.data.datasets.builtin_meta import COCO_PERSON_KEYPOINT_NAMES, COCO_PERSON_KEYPOINT_FLIP_MAP, KEYPOINT_CONNECTION_RULES
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
import cv2
import random
import time
import os
from track.predictor import Visualization

CUSTOM_DATASET_NAME = "handball_keypoints"

#register_coco_instances(CUSTOM_DATASET_NAME, {}, "./dataset/coco/annotations/person_keypoints_val2017.json", "./dataset/coco/annotations/val2017/")
#register_coco_instances(CUSTOM_DATASET_NAME, {}, "dataset/handball_other/test/000000000785.json", "./dataset/handball_other/test/")
#register_coco_instances(CUSTOM_DATASET_NAME, {}, "dataset/handball.json", "./dataset/handball/")
#register_coco_instances(CUSTOM_DATASET_NAME, {}, "dataset/handball_full_hd.json", "./dataset/handball/full_hd/")
register_coco_instances(CUSTOM_DATASET_NAME, {}, "dataset/handball.json", "./dataset/handball/full_hd/")

handball_metadata = MetadataCatalog.get(CUSTOM_DATASET_NAME)
handball_metadata.set(thing_classes=["person"])
handball_metadata.set(keypoint_names=COCO_PERSON_KEYPOINT_NAMES)
handball_metadata.set(keypoint_flip_map=COCO_PERSON_KEYPOINT_FLIP_MAP)
handball_metadata.set(keypoint_connection_rules=KEYPOINT_CONNECTION_RULES)

dataset_dicts = DatasetCatalog.get(CUSTOM_DATASET_NAME)


def show_random_from_dataset(sample_size=5):
    for d in random.sample(dataset_dicts, sample_size):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=handball_metadata, scale=0.5)
        vis = visualizer.draw_dataset_dict(d)
        cv2.imshow("radnom_sample_keypoints", vis.get_image()[:, :, ::-1])
        cv2.waitKey(1)
        time.sleep(2)
    cv2.destroyAllWindows()
    cv2.waitKey(1)


def create_cfg():
    cfg = get_cfg()
    cfg.merge_from_file("./detectron2/configs/COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml")
    cfg.DATASETS.TRAIN = (CUSTOM_DATASET_NAME,)
    cfg.DATASETS.TEST = ()  # no metrics implemented for this dataset
    cfg.DATALOADER.NUM_WORKERS = 1
    cfg.MODEL.WEIGHTS = "detectron2://COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x/138363331/model_final_997cc7.pkl"  # initialize from model zoo
    #cfg.MODEL.WEIGHTS = "./output/model_final.pkl"  # initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.002 # 0.00025
    # cfg.SOLVER.MAX_ITER = (300)  # 300 iterations seems good enough, but you can certainly train longer
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (128)  # faster, and good enough for this toy dataset
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # 1 class (person)
    cfg.MODEL.DEVICE = "cpu"
    cfg.SOLVER.MAX_ITER = 200 # 300
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    return cfg


def train():

    cfg = create_cfg()

    if not os.path.exists("./output/model_final.pth"):
        import torch
        trainer = DefaultTrainer(cfg)
        torch.save(trainer.model.state_dict(), './output/model_final.pth')

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=True) # Use trained model
    trainer.train()


def predict_random_from_dataset(sample_size=5, custom_dataset=None):

    dataset = dataset_dicts if custom_dataset is None else custom_dataset

    if sample_size is not None:
        dataset = random.sample(dataset, sample_size)

    cfg = create_cfg()

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set the testing threshold for this model
    cfg.DATASETS.TEST = (CUSTOM_DATASET_NAME, )

    visualizer = Visualization(cfg)
    for d in dataset:
        im = cv2.imread(d["file_name"])
        predictions, visualized_output = visualizer.run_on_image(im)

        cv2.imshow("random_sample", visualized_output.get_image()[:, :, ::-1])
        cv2.waitKey(1)

    cv2.destroyAllWindows()
    cv2.waitKey(1)


"""
    Execute functions
"""

show_random_from_dataset()
train()
predict_random_from_dataset()

predict_random_from_dataset(custom_dataset=[{"file_name": "./dataset/handball/full_hd/5_{0}.png".format(x)} for x in [10, 20, 30, 40, 50, 60, 70, 75]], sample_size=None)


"""
# get keypoint list
poses_1 = []
for d in dataset_dicts:
    poses_1.append(d["annotations"][0]["keypoints"])
poses_2 = poses_1

poses_1 = np.array(poses_1).reshape(40, 17, 3)
poses_2 = np.array(poses_2).reshape(40, 17, 3)

predictions["instances"].get("pred_keypoints").numpy().reshape(17, 3)
"""
