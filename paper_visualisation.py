from detectron2.data.datasets.builtin_meta import COCO_PERSON_KEYPOINT_NAMES
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pose_tracker import PoseTracker
from pose_comparator import find_and_compute_distance
from pose_visualizer import PoseVisualizer


# detection and tracking example
pt = PoseTracker(video="./video/clipped_DSC_2354.MOV", model_weights=None)
#pt.run(False)
pt.load()

cap = cv2.VideoCapture(pt.video)
cap.set(cv2.CAP_PROP_POS_FRAMES, 15)

success, frame = cap.read()
predictions, detections, out_scores = pt.detect(frame)

tracked = pt.track(frame, detections, out_scores)
pt.visualizer.draw_only_keypoints = lambda x, y: x
stop = pt.show(frame)
#cv2.waitKey(1)
#cv2.imwrite("./img/detection_and_tracking.png", frame)
plt.imsave("./img/detection_and_tracking.svg", frame[:, :, ::-1])



def estimate_pose(filename, pose_estimator):
    img = cv2.imread("./dataset/handball/full_hd/{0}.png".format(filename))
    predictions, detections, out_scores = pose_estimator.detect(img)
    img = pose_estimator.visualizer.draw_only_keypoints(img, predictions["instances"])
    return img[0:800,:][:,0:300][:, :, ::-1]


pt = PoseTracker(video="./video/clipped_DSC_2354.MOV", model_weights=None)

img1 = estimate_pose("18_40", pt)
img2 = estimate_pose("18_45", pt)
img3 = estimate_pose("18_50", pt)
img4 = estimate_pose("5_40", pt)
img5 = estimate_pose("5_45", pt)
img6 = estimate_pose("5_50", pt)


fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)
#fig.suptitle('Pose estimation examples')
ax1.imshow(img1)
ax2.imshow(img2)
ax3.imshow(img3)
ax4.imshow(img4)
ax5.imshow(img5)
ax6.imshow(img6)

for i, ax in enumerate(fig.get_axes()):
    #ax.set_axis_off()
    ax.set_xticks(())
    ax.set_yticks(())
    for side in ["left", "top", "right", "bottom"]:
        ax.spines[side].set_color("red")

plt.tight_layout()
fig.set_size_inches(5.5, 7.5)

plt.savefig("./img/poses_estimation.svg")

"""
img = cv2.imread("./dataset/handball/full_hd/.png")
predictions, detections, out_scores = pt.detect(img)
img = pt.visualizer.draw_only_keypoints(img, predictions["instances"])
plt.imshow(img[0:800,:][:,0:300][:, :, ::-1], cmap="gray")
"""



pt_trained = PoseTracker(video="./video/clipped_DSC_2354.MOV")

img1_trained = estimate_pose("18_40", pt_trained)
img2_trained = estimate_pose("18_45", pt_trained)
img3_trained = estimate_pose("18_50", pt_trained)
img4_trained = estimate_pose("5_40", pt_trained)
img5_trained = estimate_pose("5_45", pt_trained)
img6_trained = estimate_pose("5_50", pt_trained)

"""
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9), (ax10, ax11, ax12)) = plt.subplots(4, 3)
ax1.imshow(img1)
ax2.imshow(img2)
ax3.imshow(img3)
ax4.imshow(img1_trained)
ax5.imshow(img2_trained)
ax6.imshow(img3_trained)
ax7.imshow(img4)
ax8.imshow(img5)
ax9.imshow(img6)
ax10.imshow(img4_trained)
ax11.imshow(img5_trained)
ax12.imshow(img6_trained)
"""
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)
ax1.imshow(img1)
ax2.imshow(img2)
ax3.imshow(img3)
ax4.imshow(img1_trained)
ax5.imshow(img2_trained)
ax6.imshow(img3_trained)

fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)
ax1.imshow(img4)
ax2.imshow(img5)
ax3.imshow(img6)
ax4.imshow(img4_trained)
ax5.imshow(img5_trained)
ax6.imshow(img6_trained)

for i, ax in enumerate(fig.get_axes()):
    ax.set_xticks(())
    ax.set_yticks(())
    #color = "green" if i+1 in [4,5,6,10,11,12] else "red"
    color = "green" if i not in range(0, 3) else "red"
    for side in ["left", "top", "right", "bottom"]:
        ax.spines[side].set_color(color)

plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace = 0)
plt.tight_layout(0.2)
plt.margins(0, 0)
#fig.set_size_inches(2.5, 8)
fig.set_size_inches(4, 6.2)

# Stick figure estimation improvment after training, left shows estimation before training while right shows estimation after training
#plt.savefig("./img/old_detection_vs_new_detection.svg")
plt.savefig("./img/old_detection_vs_new_detection_1.svg")
plt.savefig("./img/old_detection_vs_new_detection_2.svg")






pt_t = PoseTracker(video="./video/clipped_DSC_2354.MOV")
pt_t.load()

#pt_t.frame_id = 30
pt_t.frame_id = 35
cap = cv2.VideoCapture(pt_t.video)
cap.set(cv2.CAP_PROP_POS_FRAMES, pt_t.frame_id)

success, frame = cap.read()

draw_only_kp_fnc = pt_t.visualizer.draw_only_keypoints
pt_t.visualizer.draw_only_keypoints = lambda x, y: x
img = np.array(frame[:, :, ::-1])
frame = pt_t.add_boxes(frame, pt_t.frame_id)
frame = frame[:, :, ::-1]
#plt.imshow(frame)

target = pt_t._find_tracker_by_id(pt_t.dets[pt_t.frame_id], 7)

c_id, c_x0, c_y0, c_w, c_h, c_conf = target["tracker"]
c_x0, c_y0, c_w, c_h = pt_t.add_padding((c_x0, c_y0, c_w, c_h))
curr_bbox = img[c_y0:c_y0+c_h, c_x0:c_x0+c_w]

t_curr_bbox = pt_t.aligner.resize(curr_bbox)

tmp_img = np.zeros_like(img)
tmp_img[0:t_curr_bbox.shape[0]][:, 0:t_curr_bbox.shape[1]] = t_curr_bbox
t_predictions, t_detections, t_out_scores = pt_t.detect(tmp_img)

t_img = tmp_img[0:t_curr_bbox.shape[0]][:, 0:t_curr_bbox.shape[1]]
t_pose = draw_only_kp_fnc(np.zeros_like(t_img), t_predictions["instances"])

#plt.imshow(frame)
#plt.imshow(t_img)
#plt.imshow(t_pose)

n_frame = np.array(frame)
n_frame[270:1070, 900:1200] = t_img
n_frame[270:1070, 1400:1700] = t_pose

plt.imshow(n_frame)

rect = patches.Rectangle((900, 1070), 300, -800, linewidth=2, edgecolor='red', facecolor='none')
plt.gca().add_patch(rect)

rect = patches.Rectangle((1400, 1070), 300, -800, linewidth=2, edgecolor='orange', facecolor='none')
plt.gca().add_patch(rect)

plt.plot([c_x0, 900], [c_y0, 270], color="red")
plt.plot([c_x0, 900], [c_y0+c_h, 270+800], color="red")
plt.plot([c_x0+c_w, 900+300-300], [c_y0+c_h, 270+800-200], color="red")
plt.plot([c_x0+c_w, 900], [c_y0, 270+50], color="red")

plt.arrow(1210, 270+400, 150, 0, color="orange", head_width=30, head_length=30, width=5)

plt.gca().set_axis_off()

plt.tight_layout(0.1)
plt.margins(0, 0)
plt.gcf().set_size_inches(6.4, 3.6)

#plt.savefig("./img/poses_croped_and_estimated_1.svg")
plt.savefig("./img/poses_croped_and_estimated_2.svg")


"""
f1_img = plt.imread("./img/poses_croped_and_estimated_1.png")
f2_img = plt.imread("./img/poses_croped_and_estimated_2.png")
fig, ((ax1), (ax2)) = plt.subplots(2, 1)
ax1.imshow(f1_img)
ax2.imshow(f2_img)

for i, ax in enumerate(fig.get_axes()):
    ax.set_axis_off()
plt.tight_layout()

# Few frames of tracked player executing a jump shop where stick figures are estimated after performing necessary transformation to the detected portion of the image
plt.savefig("./img/poses_croped_and_estimated.svg")
"""






pt_trained = PoseTracker(video="./video/clipped_DSC_2354.MOV")
pt_trained.load()
pt_trained.pose_visualizer.plot3D(7)
plt.tight_layout(0)
#plt.gca().view_init(elev=40, azim=40)
plt.gcf().set_size_inches(12, 6.5)
#plt.margins(0, 0, 0)

# 3D visualisation of joints in space and time when executing a jump shot
plt.savefig("./img/poses3D.svg")






pt = PoseTracker(video="./video/clipped_DSC_2354.MOV")
pt.load()

pt2 = PoseTracker(video="./video/clipped_2_DSC_2354.MOV")
pt2.load()

poses_1 = pt.pose_visualizer.poses_for_id(7)
poses_2 = pt2.pose_visualizer.poses_for_id(3)

#poses_1, poses_2, distances = find_and_compute_distance(poses_1, poses_2)
#poses_1 = poses_1[:37]
#poses_2 = poses_2[:37]
#PoseVisualizer.show_sequence(poses_1, poses_2, delay=0)

p_1 = poses_1[10]
p_2 = poses_2[10][:, :3] + [-30,100,0]
p1 = PoseVisualizer.draw_pose(p_1, "Pose")
p2 = PoseVisualizer.draw_pose(p_2, "Pose")

p2_transparent = np.array(p2)
p2_transparent = np.concatenate((p2_transparent[:, :, ::-1], np.repeat(255, 800*300).reshape(800, 300, 1)), axis=2)
for i in range(300):
    for j in range(800):
        if np.all(p2_transparent[j, i, :3] == np.array([0,0,0])):
            p2_transparent[j, i, :] = 0


lhi = np.where(np.array(COCO_PERSON_KEYPOINT_NAMES) == "left_hip")[0][0]
rhi = np.where(np.array(COCO_PERSON_KEYPOINT_NAMES) == "right_hip")[0][0]
def find_midpoint(p1, p2):
    return np.array([(p1[0]+p2[0])/2, (p1[1]+p2[1])/2])

p_1_m = find_midpoint(p_1[lhi], p_1[rhi])
p_2_m = find_midpoint(p_2[lhi], p_2[rhi])

t = p_1_m - p_2_m

p_2_aligned = np.array(p_2)
p_2_aligned[:, :2] += t
p2_aligned = PoseVisualizer.draw_pose(p_2_aligned, "Pose")
p_2_aligned_m = find_midpoint(p_2_aligned[lhi], p_2_aligned[rhi])

p2_aligned_correct = PoseVisualizer.draw_pose(p_2_aligned, "Pose", p_1)

fig, ((ax1, ax2, ax3, ax4)) = plt.subplots(1, 4)
ax1.imshow(p1[:, :, ::-1])
ax1.scatter(p_1_m[0], p_1_m[1], color="orange", s=15, marker="s")
ax2.imshow(p2[:, :, ::-1])
ax2.scatter(p_2_m[0], p_2_m[1], color="orange", s=15, marker="s")

#ax2.arrow(p_2_m[0], p_2_m[1], t[0], t[1], color="orange", linestyle="--")#, head_width=15, head_length=15)
#ax2.imshow(p2_aligned[:, :, ::-1], alpha=.3)
#ax2.scatter(p_2_aligned_m[0], p_2_aligned_m[1], color="orange", s=15, marker="s", alpha=.3)

ax3.imshow(p2_aligned[:, :, ::-1])
ax3.imshow(p2_transparent, alpha=.5)
ax3.arrow(p_2_m[0], p_2_m[1], t[0], t[1], color="orange", linestyle="--", alpha=.4)#, head_width=15, head_length=15)
ax3.scatter(p_2_aligned_m[0], p_2_aligned_m[1], color="orange", s=15, marker="s")
ax3.scatter(p_2_m[0], p_2_m[1], color="orange", s=15, marker="s", alpha=.4)

ax4.imshow(p2_aligned_correct[:, :, ::-1])

for i, ax in enumerate(fig.get_axes()):
    ax.set_xticks(())
    ax.set_yticks(())
    color = "blue" if i == 0 else "red"
    for side in ["left", "top", "right", "bottom"]:
        ax.spines[side].set_color(color)

plt.tight_layout(0.1)
plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace = 0)

plt.gcf().set_size_inches(7.8, 5)
# Aligning poses based on the points which are calculated as midpoint between hips of the to poses.
# After aligning the poses we can vizualize the difference in second pose compare to first one
plt.savefig("./img/stick_figures_aligned.svg")





pt = PoseTracker(video="./video/clipped_DSC_2354.MOV")
pt.load()
pt2 = PoseTracker(video="./video/clipped_2_DSC_2354.MOV")
pt2.load()
poses_1 = pt.pose_visualizer.poses_for_id(7)
poses_2 = pt2.pose_visualizer.poses_for_id(3)

p1 = PoseVisualizer.draw_pose(p_1, "Pose")
p2 = PoseVisualizer.draw_pose(p_2, "Pose")

p1_seq = [PoseVisualizer.draw_pose(p, "Pose")[:, :, ::-1] for p in poses_1[:20]]
p2_seq = [PoseVisualizer.draw_pose(p, "Pose")[:, :, ::-1] for p in poses_2[:30]]

img_s1 = np.array(np.repeat(255, 800*300*5*3)).reshape((800, 300*5, 3))

for p in p1_seq:
    img_s1 = np.concatenate((img_s1, p), axis=1)

img_s1 = np.concatenate((img_s1, np.array(np.repeat(255, 800*300*5*3)).reshape((800, 300*5, 3))), axis=1)
img_s1 = img_s1.astype(int)


img_s2 = np.array([]).reshape((800, 0, 3))
for p in p2_seq:
    img_s2 = np.concatenate((img_s2, p), axis=1)

img_s2 = img_s2.astype(int)


f_img = np.concatenate((img_s1, np.array(np.repeat(255, 200*300*30*3)).reshape((200, 300*30, 3)), img_s2), axis=0)
plt.imshow(f_img)
plt.vlines(5*300, 0, 800*2+200,  color="orange")
plt.vlines(25*300, 0, 800*2+200,  color="orange")
plt.arrow(5*300, 800+150, 20*300-50, 0,  color="orange", linestyle="--", head_width=50, head_length=50)
plt.arrow(25*300, 800+150, -(20*300-50), 0,  color="orange", linestyle="--", head_width=50, head_length=50)
plt.text(11*300, 800+140, s="Search for window with lowest distance", fontsize=12, color="orange")
plt.gcf().get_axes()[0].set_axis_off()
plt.tight_layout(0.)
plt.gcf().set_size_inches(14.5, 3)

# Finding appropriate sequence of poses $S2$ by finding a window with minimal distance from $S1$
plt.savefig("./img/window_sequence.svg")



cv2.destroyAllWindows()
cv2.waitKey(1)
