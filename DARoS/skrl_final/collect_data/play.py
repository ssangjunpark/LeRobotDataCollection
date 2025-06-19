import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="playing diffusion policy")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import io
import os
import torch
import matplotlib.pyplot as plt

import omni

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.manager_based.locomotion.velocity.config.h1.rough_env_cfg import H1RoughEnvCfg_PLAY
from isaaclab_tasks.manager_based.DARoS.multidoorman.multidoorman_env_cfg import MultidoormanEnvCfg_PLAY, MultidoormanCameraEnvCfg_PLAY

from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy

def main():
    policy = DiffusionPolicy.from_pretrained('/home/isaac/Documents/Github/lerobot/PolicyFromIsaac/outputs/train/2025-06-18 02:22:46.394533')

    device = "cuda"

    env_cfg = MultidoormanEnvCfg_PLAY()
    env_cfg.scene.num_envs = 1
    env_cfg.curriculum = None
    env_cfg.sim.device = args_cli.device

    if args_cli.device == "cpu":
        env_cfg.sim.use_fabric = False

    env = ManagerBasedRLEnv(cfg=env_cfg)

    # print(policy.config.input_features) # {'observation.images.top': PolicyFeature(type=<FeatureType.VISUAL: 'VISUAL'>, shape=(3, 256, 256)), 'observation.images.hand1': PolicyFeature(type=<FeatureType.VISUAL: 'VISUAL'>, shape=(3, 256, 256)), 'observation.images.hand2': PolicyFeature(type=<FeatureType.VISUAL: 'VISUAL'>, shape=(3, 256, 256)), 'observation.state': PolicyFeature(type=<FeatureType.STATE: 'STATE'>, shape=(45,))}
    # print(env.observation_space) # Dict('policy': Box(-inf, inf, (1, 45), float32))
    # print(policy.config.output_features) # {'action': PolicyFeature(type=<FeatureType.ACTION: 'ACTION'>, shape=(13,))}
    # print(env.action_space) # Box(-inf, inf, (1, 13), float32)
    # exit()

    scene = env.scene

    # print(scene['tiled_camera1'].data.output["rgb"].shape)
    # plt.imshow(scene['tiled_camera1'].data.output["rgb"][0].cpu().numpy())
    # plt.title("cam1")
    # plt.show()

    # print(scene['tiled_camera2'].data.output["rgb"].shape)
    # plt.imshow(scene['tiled_camera2'].data.output["rgb"][0].cpu().numpy())
    # plt.title("cam2")
    # plt.show()

    # print(scene['tiled_camera3'].data.output["rgb"].shape)
    # plt.imshow(scene['tiled_camera3'].data.output["rgb"][0].cpu().numpy())
    # plt.title("cam3")
    # plt.show()

    # run inference with the policy
    obs, _ = env.reset()
    with torch.inference_mode():
        while simulation_app.is_running():

            cam_data = [scene['tiled_camera1'].data.output["rgb"][0].to(torch.float32).permute(2, 0, 1) / 255, 
                        scene['tiled_camera2'].data.output["rgb"][0].to(torch.float32).permute(2, 0, 1) / 255,
                        scene['tiled_camera3'].data.output["rgb"][0].to(torch.float32).permute(2, 0, 1) / 255]
            #print(obs)
            obs = obs['policy'].to(torch.float32).to(device, non_blocking=True)

            observation = {
                'observation.images.top' : cam_data[0].to(device, non_blocking=True).unsqueeze(0),
                'observation.images.hand1' : cam_data[1].to(device, non_blocking=True).unsqueeze(0),
                'observation.images.hand2' : cam_data[2].to(device, non_blocking=True).unsqueeze(0),
                'observation.state' : obs,
            }

            action = policy.select_action(observation)
            print(action)
            print(action.shape)
            obs, _, _, _, _ = env.step(action)

if __name__ == "__main__":
    main()
    simulation_app.close()