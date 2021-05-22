# MCL-SLAM-for-THOR
SLAM for the THOR-OP Humanoid robot

The data consists of Planar Lidar scans of the THOR-OP Humanoid robot along with all the joint angles. The lidar scans are projected onto the map coordinates and is filtered so that the ground itself is not detected as an obstacle. Particle filter approach is used to localise the robot and updation of the log-odds map is done using the lidar scan of the particle with the maximum correlation. 


## Results of SLAM

The final map and an animation of it's creation is given below.
### Map 1

<img src = "https://github.com/vashist123/MCL-SLAM-for-THOR/blob/main/results/map_0.jpg" width ="400" height = "300"/> <img src="https://github.com/vashist123/MCL-SLAM-for-THOR/blob/main/map_0.gif" width ="400" height = "300" />

### Map 2

<img src = "https://github.com/vashist123/MCL-SLAM-for-THOR/blob/main/results/map_1.jpg" width ="400" height = "300"> <img src="https://user-images.githubusercontent.com/68932319/119015709-28ab0a00-b967-11eb-94e9-a3f5a3c93409.gif" width ="400" height = "300" />

### Map 3

<img src = "https://github.com/vashist123/MCL-SLAM-for-THOR/blob/main/results/map_2.jpg" width ="400" height = "300"> <img src="https://github.com/vashist123/MCL-SLAM-for-THOR/blob/main/map_2.gif" width ="400" height = "300"/>

### Map 4

<img src = "https://github.com/vashist123/MCL-SLAM-for-THOR/blob/main/results/map_3.jpg" width ="400" height = "300"> <img src="https://github.com/vashist123/MCL-SLAM-for-THOR/blob/main/map_3.gif" width ="400" height = "300"/>
