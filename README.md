# PoseTracking
 Pose tracking and comparison in the video sequence


### Instalation
- TODO:


### Pose tracking

Steps in pose tracking:
1. Detect object within an image
2. Apply DeepSORT to track the detected objects
3. Crop and resize the tracked object
4. Run pose estimation on resulted image


#### Detection and tracking
We run object detection and tracking on the video sequence using Faster R-CNN and Deep SORT
![](./img/detection_and_tracking.svg)


#### Pose estimation
For each frame we crop and resize the tracked player to dimenisions 300x800, when a pose estimation is executed
![](./img/poses_croped_and_estimated.svg)


#### Pose tracking
Estimated pose for tracked player is saved, and at the end we can visualize poses across frames
![](./img/poses3D.svg)


#### Pose alignment
We can compare poses of two different players executing the same action sequence, where we observe the first one as a template and evaluate the second one to find needed correction in order to correctly execute the action
![](./img/stick_figures_aligned.svg)



### Run on a custom video sequence

#### Pose tracking

Defining a pose tracker for a video in a given path:
```python
pt = PoseTracker(video="my_video.mov")
```


We can run pose tracking for all players in the scene:
```python
pt.run()
```

Or we can run pose tracking for only one specific player, here you need to hit the key _**s**_ to enter **tracker id** of player you wish to track:
```python
pt.select_and_run()
```

Results can be saved and loaded:
```python
pt.save()
pt.load()
```


After running pose tracking we can visualize poses in space across time for a specific player:
```python
pt.pose_visualizer.plot3D(7)
```


#### Pose comparison

We can compare two pose sequences, first, we run or load tracked poses

```python
pt1 = PoseTracker(video="my_video_1.mov")
pt1.load()

pt2 = PoseTracker(video="my_video_2.mov")
pt2.load()
```

Then we need to get the specific pose sequences for a target player from each pose tracker that will be compared
```python
poses_1 = pt.pose_visualizer.poses_for_id(7)
poses_2 = pt2.pose_visualizer.poses_for_id(3)
```

We then compute the distance between the two poses and find the appropriate starting point of the second sequence to match the first one.
**Important** to notice here is that the second sequence must be longer than the first one and the function `find_and_compute_distance` will take care of the rest
```python
from pose_comparator import find_and_compute_distance

poses_1, poses_2, distances = find_and_compute_distance(poses_1, poses_2)

print("Total distance: " + str(np.array(distances).flatten().sum()))
```


Finally, we can visualize the difference (delay represents the number of seconds before continuing to the next pose)
```python
PoseVisualizer.show_sequence(poses_1, poses_2, delay=4)
```



**Final code**
```python
from pose_comparator import find_and_compute_distance

pt1 = PoseTracker(video="my_video_1.mov")
pt1.load()

pt2 = PoseTracker(video="my_video_2.mov")
pt2.load()

poses_1 = pt.pose_visualizer.poses_for_id(7)
poses_2 = pt2.pose_visualizer.poses_for_id(3)

poses_1, poses_2, distances = find_and_compute_distance(poses_1, poses_2)

print("Total distance: " + str(np.array(distances).flatten().sum()))

PoseVisualizer.show_sequence(poses_1, poses_2, delay=4)
```
