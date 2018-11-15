# Imitation Learning

This repository is basically follow the [Berkeley CS 294: Deep Reinforcement Learning](http://rll.berkeley.edu/deeprlcourse/). first assignment.

Dependencies: TensorFlow, MuJoCo version 1.31, OpenAI Gym

**Note**: MuJoCo versions until 1.5 do not support NVMe disks therefore won't be compatible with recent Mac machines.
There is a request for OpenAI to support it that can be followed [here](https://github.com/openai/gym/issues/638).

In `experts/`, the provided expert policies are:
* Ant-v1.pkl
* HalfCheetah-v1.pkl
* Hopper-v1.pkl
* Humanoid-v1.pkl
* Reacher-v1.pkl
* Walker2d-v1.pkl

The name of the pickle file corresponds to the name of the gym environment.

## Set up on ubuntu

* download getid executable file, from [MuJoco](https://www.roboti.us/license.html)
* run the getid file
	+ `sudo chmod +x getid_linux`
	+ `./getid_linux`
* register and follow the instruction on email (download 131, put in ~/.mujoco/mjpro131, see [link](https://github.com/openai/mujoco-py))
* create conda environment
	+ `conda create -n bc python=3.5 numpy scipy matplotlib theano keras ipython jupyter scikit-learn`
	+ `source activate bc`
* install mujoco-py 0.5.7, following the [link](https://github.com/openai/mujoco-py)
* install gym: 
	+ `pip install gym==0.7.4`

## Train

check `bc_train.py` file to see how to define bc model and train it on expert data.
Example usage:
* `python bc_train.py Humanoid-v1 --render --train_steps 20000 --num_rollouts 20`

## Evaluation

check `bc_eval.py` file to see how to evaluate trained model.
Example usage:
* `python bc_eval.py Humanoid-v1 --render --num_rollouts 20`

