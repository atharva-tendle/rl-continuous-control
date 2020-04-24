# rl-continuous-control

## Introduction

This project was made as a part of the Deep Reinforcement Learning Nanodegree. 
A DDPG agent is trained to operate a double jointed arm. The goal is to reach an object. Unity environment is used for training. 
The code is written in Python 3 and Pytorch.

![Agent](resources/reacher.gif)

## Installation
- First follow the instructions on the drlnd github page to setup the required packages and modules: [prerequisites](https://github.com/udacity/deep-reinforcement-learning/#dependencies).
- For this project, you will not need to install Unity - this is because we have already built the environment for you, and you can download it from one of the links below. You need only select the environment that matches your operating system:

**Version 1: One (1) Agent**

- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

**Version 2: Twenty (20) Agents**

- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

- (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/15056/windows-32-64-bit-faq) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

- (_For AWS_) If you'd like to train the agent on AWS (and have not enabled a [virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the "headless" version of the environment. You will not be able to watch the agent without enabling a virtual screen, but you will be able to train the agent. (To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the Linux operating system above.)

## Environment
- In this environment, a double-jointed arm can move to target locations. 
- A reward of +0.1 is provided for each step that the agent's hand is in the goal location. 
- Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.
- The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. 
- Each action is a vector with four numbers, corresponding to torque applicable to two joints. 
- Every entry in the action vector should be a number between -1 and 1.
- For this project, we will provide you with two separate versions of the Unity environment:
    - The first version contains a single agent.
    - The second version contains 20 identical agents, each with its own copy of the environment.

## Solving the Environment
**Option 1: Solve the First Version**

- The task is episodic, and in order to solve the environment, your agent must get an average score of +30 over 100 consecutive episodes.

**Option 2: Solve the Second Version**

- The barrier for solving the second version of the environment is slightly different, to take into account the presence of many agents. - - In particular, your agents must get an average score of +30 (over 100 consecutive episodes, and over all agents). 
- Specifically, after each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 20 (potentially different) scores. We then take the average of these 20 scores.
- This yields an average score for each episode (where the average is over all 20 agents).

