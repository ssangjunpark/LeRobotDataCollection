"""
outputs: (tensor([[-1.4079,  0.4560,  0.6863,  0.5787, -0.5894, -1.3362,  1.0691,  1.6639,
          0.1701,  0.6352, -0.3048, -0.4061,  0.7830]], device='cuda:0'), tensor([[-18.5473]], device='cuda:0'), {'mean_actions': tensor([[-0.2928, -0.0228,  0.7302, -0.0603, -0.8046, -0.0378, -0.0160, -0.7734,
         -0.0925,  0.2593, -0.2734, -0.0530, -0.9060]], device='cuda:0')})
outputs type: <class 'tuple'>
action: tensor([[-0.2928, -0.0228,  0.7302, -0.0603, -0.8046, -0.0378, -0.0160, -0.7734,
         -0.0925,  0.2593, -0.2734, -0.0530, -0.9060]], device='cuda:0')
action type: <class 'torch.Tensor'>
observation: tensor([[-9.1644e-02,  4.6365e-02,  4.2678e-01, -8.6003e-02, -3.0993e-01,
         -5.9141e-02, -2.7520e-01, -3.1952e-02, -6.5647e-02,  2.1288e-01,
         -4.2869e-01,  2.0686e-01,  1.5160e-01,  1.4507e-01, -3.1337e-01,
         -1.5929e-01,  9.3503e-04, -4.3486e-02, -3.5802e-02, -3.8278e-01,
          0.0000e+00,  1.5000e+00,  1.0000e+00,  7.0700e-01, -2.0706e-08,
          6.1395e-09,  7.0700e-01,  0.0000e+00,  1.5000e+00,  1.0000e+00,
          8.9938e-31,  1.2542e-22, -2.9282e-01, -2.2796e-02,  7.3025e-01,
         -6.0268e-02, -8.0459e-01, -3.7767e-02, -1.5977e-02, -7.7344e-01,
         -9.2497e-02,  2.5927e-01, -2.7342e-01, -5.2997e-02, -9.0596e-01]],
       device='cuda:0')
observation type: <class 'torch.Tensor'>
"""


# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to play a checkpoint of an RL agent from skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["AMP", "PPO", "IPPO", "MAPPO"],
    help="The RL algorithm used for training the skrl agent.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

import skrl
from packaging import version

# check for minimum supported skrl version
SKRL_VERSION = "1.4.2"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg

# PLACEHOLDER: Extension template (do not remove this comment)

# config shortcuts
algorithm = args_cli.algorithm.lower()

from DataRecoder import DataRecoder

import matplotlib.pyplot as plt


def main():
    """Play with skrl agent."""
    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    task_name = args_cli.task.split(":")[-1]

    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )

    try:
        experiment_cfg = load_cfg_from_registry(task_name, f"skrl_{algorithm}_cfg_entry_point")
    except ValueError:
        experiment_cfg = load_cfg_from_registry(task_name, "skrl_cfg_entry_point")

    # specify directory for logging experiments (load checkpoint)
    log_root_path = os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # get checkpoint path
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("skrl", task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            log_root_path, run_dir=f".*_{algorithm}_{args_cli.ml_framework}", other_dirs=["checkpoints"]
        )
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # get environment (step) dt for real-time evaluation
    try:
        dt = env.step_dt
    except AttributeError:
        dt = env.unwrapped.step_dt

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)  # same as: `wrap_env(env, wrapper="auto")`
    
    
    scene = env._unwrapped.scene
    
    # configure and instantiate the skrl runner
    # https://skrl.readthedocs.io/en/latest/api/utils/runner.html
    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0  # don't log to TensorBoard
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0  # don't generate checkpoints
    runner = Runner(env, experiment_cfg)

    data_recorder = DataRecoder(dt = env.physics_dt)

    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    runner.agent.load(resume_path)
    # set agent to evaluation mode
    runner.agent.set_running_mode("eval")

    # reset environment
    obs, _ = env.reset()
    timestep = 0
    
    num_episodes = 1000
    curr_episode = 0
    image_size = 256*256*3
    res = False
    term = False
    num_collected = 0

    
    with torch.inference_mode():
        while simulation_app.is_running():
            print(f"Inferencing {curr_episode+1}/{num_episodes}")
            while curr_episode < num_episodes:
                if not (res or term):
                    outputs = runner.agent.act(obs, timestep=0, timesteps=0)

                    if hasattr(env, "possible_agents"):
                        actions = {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in env.possible_agents}
                    else:
                        actions = outputs[-1].get("mean_actions", outputs[0])

                    #print(str(curr_episode) + " " + str(actions))

                    new_obs, rew, term, res, _ = env.step(actions)
                    # print(scene['top_camera'].data.output["rgb"].shape)
                    # plt.imshow(scene['top_camera'].data.output["rgb"][0].cpu().numpy())
                    # plt.title("top_camera")
                    # plt.show()

                    # print(scene['left_camera'].data.output["rgb"].shape)
                    # plt.imshow(scene['left_camera'].data.output["rgb"][0].cpu().numpy())
                    # plt.title("left_camera")
                    # plt.show()

                    # print(scene['right_camera'].data.output["rgb"].shape)
                    # plt.imshow(scene['right_camera'].data.output["rgb"][0].cpu().numpy())
                    # plt.title("right_camera")
                    # plt.show()

                    cam_data = [scene['top_camera'].data.output["rgb"][0].cpu().numpy(), 
                                scene['left_camera'].data.output["rgb"][0].cpu().numpy(),
                                scene['right_camera'].data.output["rgb"][0].cpu().numpy()]

                    data_recorder.write_data_to_buffer(observation=obs, action=actions, reward=rew, 
                                                       termination_flag=(res or term), cam_data=cam_data, 
                                                       debug_stuff=[env.max_episode_length, env.episode_length_buf.cpu().item()],
                                                       image_size=image_size)
                    
                    obs = new_obs
                else:
                    curr_episode += 1
                    print('terminated: ', term)
                    print('truncated: ', res)

                    if term:
                        data_recorder.dump_buffer_data()
                        
                        print(f"Sucess! Writing Data. {num_collected + 1} collected.")
                        num_collected += 1

                    obs, _ = env.reset()
                    data_recorder.reset()
                    res = False
                    term = False
                    print(f"Inferencing {curr_episode+1}/{num_episodes}")
                    

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
