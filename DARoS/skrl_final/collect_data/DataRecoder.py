import os
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import io
from PIL import Image

import matplotlib.pyplot as plt

SAVE_DIR = os.getcwd() + "/LeRobotData/"

class DataRecoder:
    def __init__(self, dt):
        self.log_dir = SAVE_DIR + "data/chunk_000/"
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # episode index
        self.episode_index = 0
        self.index = 0

        self.dt = dt

        self.reset()
    
    def reset(self):
        self.df = pd.DataFrame(columns=['observation.images.top_camera', 'observation.images.left_camera','observation.images.right_camera','observation.state', 'action', 'timestamp', 'episode_index', 'frame_index', 'index', 'next.reward', 'next.done', 'task_index'])
        
        self.timestamp = 0
        self.frame_index = 0

        # this is for data logging 
        self.column_index = 0

    def write_data_to_buffer(self, observation, action, reward, termination_flag, cam_data, debug_stuff, image_size):
        im_dict_to_be_added = []

        for idx in range(len(cam_data)):
            # plt.imshow(cam_data[idx].astype(np.uint8))
            # plt.title(f"{idx}")
            # plt.show()

            pil_img_top_trans = Image.fromarray(cam_data[idx].astype(np.uint8), mode="RGB")

            img_buf = io.BytesIO()
            pil_img_top_trans.save(img_buf, format="PNG")
            img_binary = img_buf.getvalue()

            img_file_name = self._name_helper('frame', '.png', self.frame_index)

            img_dict = {
                'bytes' : img_binary,
                'path' : img_file_name
            }

            im_dict_to_be_added.append(img_dict)

        #print(termination_flag)
        # save it into local memory 

        # https://docs.phospho.ai/learn/lerobot-dataset
        # LeRobot want their .parquet to have:
        # observation.state, action, timestamp, episode_index, frame_index, index, next.done(optional), task_index(optional)
        # we can also include next.reward and next.done it seems like
        #observation['policy'].cpu().numpy()[0]
        self.df.loc[self.column_index] = [im_dict_to_be_added[0], im_dict_to_be_added[1], im_dict_to_be_added[2], 
                                          observation.cpu().numpy()[0], action.cpu().numpy()[0], self.timestamp, 
                                          self.episode_index, self.frame_index, self.index, reward.cpu().item(), 
                                          termination_flag.cpu().item(), 0]
        self.column_index += 1
        self.timestamp += self.dt
        self.frame_index += 1
        self.index += 1
        im_dict_to_be_added.clear()

        #print(self.df)
        # exit()

    def dump_buffer_data(self):
        data_file_name = self._name_helper('episode', '.parquet', self.episode_index)

        self._update_log_dir()
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        table = pa.Table.from_pandas(self.df)
        pq.write_table(table, self.log_dir + data_file_name)

        self.episode_index += 1
        print(f"Complete Writing Data. Saved to {self.log_dir + data_file_name}")

    def _update_log_dir(self):
        if self.episode_index < 1000:
            self.log_dir = SAVE_DIR + "data/chunk_000/"
        elif 1000 <= self.episode_index < 2000:
            self.log_dir = SAVE_DIR + "data/chunk_001/"
        elif 2000 <= self.episode_index < 3000:
            self.log_dir = SAVE_DIR + "data/chunk_002/"
        elif 3000 <= self.episode_index < 4000:
            self.log_dir = SAVE_DIR + "data/chunk_003/"
        else:
            self.log_dir = SAVE_DIR + "data/chunk_004/"

    def _name_helper(self, name, extension, index):
        if index <= 9:
            file_name = f'{str(name)}_00000' + str(index) + str(extension)
        elif 9 < index <= 99:
            file_name = f'{str(name)}_0000' + str(index) + str(extension)
        elif 99 < index <= 999:
            file_name = f'{str(name)}_000' + str(index) + str(extension)
        elif 999 < index <= 9999:
            file_name = f'{str(name)}_00' + str(index) + str(extension)
        else:
            file_name = f'{str(name)}_0' + str(index) + str(extension)

        return file_name