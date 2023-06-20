# UR10-Reach-RL
Solving the Reach Problem for the UR10 Robot Arm Using Reinforcement Learning. In the reach problem, the robot should be able to
reach a random target position (of course, in its reachability space). The start position could be random or fixed. Here we solve for the fixed position.

Two RL algorithms are supported (--model param): 
* PPO
* DDPG
  
Two types of environments are supported (--space param): 
* In the first one (cube) the target location is restricted by the parallelepiped in front of the robot
* In the second one (sphere) the target location is restricted by a truncated sphere around the robot

In fact, this problem can be solved in two ways: when the inverse kinematics (IK) problem has already been solved (you can simply set the coordinates for the effector) or when you control the robot's motors directly. The last one is more complex.

By default, we try to solve the complex variant (without IK). But it is also possible to solve with IK (--complex param).

The code for working with the Pybullet environment is based on https://github.com/dmitrySorokin/ur10_robot/blob/master/ur10_env.py and panda-gym.

# How to run

For train: main.py --train --space cube -m ddpg
For test: main.py --test -p './models/best_model-ddpg-6-cube-099.zip' --space cube --model ddpg --evals 100


# Useful articles
1.	Guo Z. et al. A reinforcement learning approach for inverse kinematics of arm robot //Proceedings of the 2019 4th International Conference on Robotics, Control and Automation. – 2019. – С. 95-99.
2.	Zeng R. et al. Manipulator control method based on deep reinforcement learning //2020 Chinese Control And Decision Conference (CCDC). – IEEE, 2020. – С. 415-420.
3.	Weber J., Schmidt M. An improved approach for inverse kinematics and motion planning of an industrial robot manipulator with reinforcement learning //2021 Fifth IEEE International Conference on Robotic Computing (IRC). – IEEE, 2021. – С. 10-17.
4.	Zhang Z., Zheng C. Simulation of robotic arm grasping control based on proximal policy optimization algorithm //Journal of Physics: Conference Series. – IOP Publishing, 2022. – Т. 2203. – №. 1. – С. 012065.
