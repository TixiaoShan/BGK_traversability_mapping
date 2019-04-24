# Bayesian Generalized Kernel Inference for Terrain Traversability Mapping

This repository contains code for the paper "Bayesian Generalized Kernel Inference for Terrain Traversability Mapping".


## Get Started

- Install [ROS](http://www.ros.org/install/).

## Compile

You can use the following commands to download and compile the package.

```
cd ~/catkin_ws/src
git clone https://github.com/tixiaoshan/BGK_traversability_mapping.git
cd ..
catkin_make
```

## Run the System (in simulation)

1. Run the launch file:
```
roslaunch bgk_traversability_mapping run.launch
```

2. Play existing bag files in this repo:
```
rosbag play *.bag --clock
```
Notes: you need to proive the TF transformation between /map and /base_link.


## Cite Our Paper

Thank you for citing our paper if you use any of this code: 
```
@inproceedings{traversability2018,
  title={Bayesian Generalized Kernel Inference for Terrain Traversability Mapping},
  author={Tixiao Shan, Kevin Doherty, Jinkun Wang and Brendan Englot},
  booktitle={In Proceedings of the 2nd Annual Conference on Robot Learning},
  year={2018}
}
```
