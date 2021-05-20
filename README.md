# MCL-SLAM-for-THOR
SLAM for the THOR-OP Humanoid robot

The data consists of Planar Lidar scans of the THOR-OP Humanoid robot along with all the joint angles. The lidar scans are projected onto the map coordinates and is filtered so that the ground itself is not detected as an obstacle. Particle filter approach is used to localise the robot and updation of the log-odds map is done using the lidar scan of the particle with the maximum correlation.

## Results of SLAM
### Map 1

<img src = "https://github.com/vashist123/MCL-SLAM-for-THOR/blob/main/results/map_0.jpg">
![map_0](https://user-images.githubusercontent.com/68932319/119015466-eb467c80-b966-11eb-9a13-3fb03fc9d536.gif)


### Map 2

<img src = "https://github.com/vashist123/MCL-SLAM-for-THOR/blob/main/results/map_1.jpg">
![map_1](https://user-images.githubusercontent.com/68932319/119015709-28ab0a00-b967-11eb-94e9-a3f5a3c93409.gif)

### Map 3

<img src = "https://github.com/vashist123/MCL-SLAM-for-THOR/blob/main/results/map_2.jpg">
![](https://github.com/vashist123/MCL-SLAM-for-THOR/blob/main/gifs/map_2.gif)

### Map 4

<img src = "https://github.com/vashist123/MCL-SLAM-for-THOR/blob/main/results/map_3.jpg">
![](https://github.com/vashist123/MCL-SLAM-for-THOR/blob/main/gifs/map_3.gif)
