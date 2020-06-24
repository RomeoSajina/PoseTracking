"""
https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch
https://github.com/jsbroks/coco-annotator
https://ckrobotics.wordpress.com/2018/12/31/generating-coco-style-dataset/

cd coco-annotator
docker-compose up
docker container ls

URL:
http://localhost:5000/ admin:admin

cd handball
docker cp . a03b7a7f806d:/datasets/handball # annotator_workers

# stop
docker stop $(docker ps -a -q)



Procedure:

1. Use 'clip' method to create a mini video where player images will be taken
2. Use 'create_images' to create player images based on the tracker id
3. Run coco-annotator to anotate images
    - annontate keypoints
    - annotate bbox
    - export dataset in .json
4. Run 'fix_exported_json' on exported .json file to reformat it for use with detectron

"""
from pose_tracker import PoseTracker
import cv2
import numpy as np
import json
import collections


def clip(video, output_video):
    cap = cv2.VideoCapture(video)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video, fourcc, round(cap.get(cv2.CAP_PROP_FPS)), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    start = False

    while cap.isOpened():

        success, frame = cap.read()

        if not success:
            break

        cv2.imshow("video", frame)
        k = cv2.waitKey(3)

        if k == ord("s"):
            start = True

        if k == ord("e"):
            break

        if start:
            out.write(frame)

    cap.release()
    out.release()

    cv2.destroyAllWindows()
    cv2.waitKey(1)


def create_images(o_v, v_focus_id, full_hd=True):
    #o_v, v_focus_id = o_v_1, v1_focus_id
    pt = PoseTracker(video=o_v)

    cap = cv2.VideoCapture(pt.video)
    pt.init()

    frame_id = 1

    while cap.isOpened():

        success, frame = cap.read()

        if not success:
            break

        predictions, detections, out_scores = pt.detect(frame)

        tracked = pt.track(frame, detections, out_scores)

        f_track = pt._find_tracker_by_id({"tracked": tracked}, v_focus_id)

        if f_track is not None:
            id, x0, y0, w, h, conf = f_track["tracker"]
            #_, x0, y0, w, h, conf = f_track["overlap_detections"][0]

            x0, y0, w, h = pt.add_padding((x0, y0, w, h))
            bbox_img = frame[y0:y0+h, x0:x0+w]

            bbox_img = pt.aligner.resize(bbox_img)

            # convert to 1920 x 1080
            if full_hd:
                zeros = np.zeros((1080, 1920, 3))
                zeros[0:bbox_img.shape[0]][:, 0:bbox_img.shape[1]] = bbox_img
                bbox_img = zeros

            cv2.imshow("focused_bbox_img", bbox_img)
            cv2.waitKey(1)

            if full_hd:
                cv2.imwrite("./dataset/handball/full_hd/{0}_{1}.png".format(v_focus_id, frame_id), bbox_img)
            else:
                cv2.imwrite("./dataset/handball/{0}_{1}.png".format(v_focus_id, frame_id), bbox_img)

            # result before
            tmp_img = np.zeros_like(frame)

            tmp_img[0:bbox_img.shape[0]][:, 0:bbox_img.shape[1]] = bbox_img
            # t_predictions, t_detections, t_out_scores = pt.detect(tmp_img)
            predictions, visualized_output = pt.visualizer.run_on_image(tmp_img)

            if full_hd:
                cv2.imwrite("./dataset/handball/full_hd/no_train_dets/{0}_{1}.png".format(v_focus_id, frame_id), visualized_output.get_image()[:, :, ::-1])
            else:
                cv2.imwrite("./dataset/handball/old/{0}_{1}.png".format(v_focus_id, frame_id), visualized_output.get_image()[:, :, ::-1])
            # /result before

        frame_id += 1

    cv2.destroyAllWindows()
    cv2.waitKey(1)


def fix_exported_json(file_name):

    with open(file_name) as json_file:

        data = json.load(json_file)

    grouped = collections.defaultdict(list)

    for item in data["annotations"]:
        grouped[item["image_id"]].append(item)

    final_annotations = []

    for image_id, group in grouped.items():

        main_group, sub_group = (group[0], group[1]) if group[0].get("keypoints") is not None else (group[1], group[0])

        main_group["bbox"] = sub_group["bbox"]
        main_group["segmentation"] = sub_group["segmentation"]
        main_group["width"] = 1920
        main_group["height"] = 1080

        final_annotations.append(main_group)


    final_images = []

    for img in data["images"]:
        img["path"] = img["file_name"]
        img["width"] = 1920
        img["height"] = 1080
        final_images.append(img)


    data["annotations"] = final_annotations
    data["images"] = final_images

    with open(file_name, 'w') as outfile:
        json.dump(data, outfile)


## RUN

v_1, o_v_1, v1_focus_id = "./video/DSC_2354.MOV", "./video/clipped_DSC_2354.MOV", 18
v_2, o_v_2, v2_focus_id = "./video/DSC_2354.MOV", "./video/clipped_2_DSC_2354.MOV", 5

clip(v_1, o_v_1)
clip(v_2, o_v_2)

create_images(o_v_1, v1_focus_id)
create_images(o_v_2, v2_focus_id)

fix_exported_json(file_name="dataset/handball.json")

#pt = PoseTracker(video=o_v_1)
#pt = PoseTracker(video=o_v_2)
#pt.run(track_pose=False)
#pt.do_pose_tracking(tracker_id=18)


"""
create full hd images manual
for i in range(3, 80):
    img = cv2.imread("./dataset/handball/5_{0}.png".format(i))
    zeros = np.zeros((1080, 1920, 3))
    zeros[0:img.shape[0]][:, 0:img.shape[1]] = img
    cv2.imwrite("./dataset/handball/full_hd/5_{0}.png".format(i), zeros)
"""






