import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="take polciy and create lerobot data")
parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint exported as jit.", required=True)


AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import io
import os
import torch

import omni

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.manager_based.locomotion.velocity.config.h1.rough_env_cfg import H1RoughEnvCfg_PLAY
from isaaclab_tasks.manager_based.DARoS.multidoorman.multidoorman_env_cfg import MultidoormanEnvCfg_PLAY

from scripts.DARoS.collect_data.DataRecoder import DataRecoder

def main():
    policy_path = os.path.abspath(args_cli.checkpoint)
    file_content = omni.client.read_file(policy_path)[2]
    file = io.BytesIO(memoryview(file_content).tobytes())
    policy = torch.jit.load(file, map_location=args_cli.device)

    env_cfg = H1RoughEnvCfg_PLAY()
    env_cfg.scene.num_envs = 1
    env_cfg.curriculum = None
    env_cfg.sim.device = args_cli.device

    if args_cli.device == "cpu":
        env_cfg.sim.use_fabric = False

    env = ManagerBasedRLEnv(cfg=env_cfg)

    data_recorder = DataRecoder()

    num_episodes = 1000
    curr_episode = 0
    
    # MDP
    obs, _ = env.reset()
    term = True
    while curr_episode < num_episodes:
        with torch.inference_mode():
            while simulation_app.is_running():
                if not term:
                    action = policy(obs["policy"])
                    print("Action: ", action)
                    obs, rew, term, _, _ = env.step(action)
                    print("Observation: ", obs)
                    data_recorder.write_data(observation=obs, reward=rew, termination_flag=term, cam_data=None)
                else:
                    obs, _ = env.reset()
                    data_recorder.reset()
                    curr_episode +=1

if __name__ == "__main__":
    main()
    simulation_app.close()
