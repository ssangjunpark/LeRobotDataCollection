"""

The following constants need to be manually filled out before any of the data collection scripts are executed.

"""

import os

"""
DataRecoder.py

SAVE_DIRECTORY_PATH - Path to where the Dataset should be stored after policy inference
LEROBOT_DATASET_COLUMN_HEADER - This is list of title of the features that we are going to log via DataRecoder and policy_inference.py
"""
SAVE_DIRECTORY_PATH_DATA : str = os.getcwd() + "/LeRobotData/"
LEROBOT_DATASET_COLUMN_HEADER : list= ['observation.images.top_camera', 'observation.images.left_camera','observation.images.right_camera','observation.state', 'action', 'timestamp', 'episode_index', 'frame_index', 'index', 'next.reward', 'next.done', 'task_index']

"""
policy_inference.py
"""
# This script is an modified version of skrl/play.py script thus you must use the CLI to specify the following:
# --task
# --checkpoint
# --num_envs
# --enable_cameras
# --headless
# for example, python3 scripts/DARoS/skrl_final/collect_data/policy_inference.py --checkpoint=/home/isaac/Documents/Github/IsaacLab/logs/skrl/multidoorman_grasp/trained/best_agent.pt --task=MultiDoorMan-Grasp-Log --enable_cameras --num_envs=1 --headless

"""
MetaRecoder.py

SAVE_DIRECTORY_PATH - Path to where the Meta files should be stored after meta data generation
"""
SAVE_DIRECTORY_PATH_META : str = os.getcwd() + "/LeRobotData/meta/"
# Logic for multiple tasks are not yet been implemented! please refer to initializer of MetaRecorder.py 
JOINT_NAMES = 'TODO' # place holder for now

"""
generate_meta.py

DATA_FOLDER_PATH - where data is stored (chunks containing the .parquet files)
"""
DATA_FOLDER_PATH = os.getcwd() + "/LeRobotData/" + 'data'

"""
play.py

DIFFUSION_POLICY_PATH - where weights for the diffusion policy is stored
"""
# DIFFUSION_POLICY_PATH must be edited manually at play.py