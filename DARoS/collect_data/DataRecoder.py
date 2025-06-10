import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

SAVE_DIR = os.getcwd() + "/LeRobotData/"

class DataRecoder:
    def __init__(self):
        self.log_dir = SAVE_DIR + "data/chunk_000/"
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.vid_dir_wrist1 = SAVE_DIR + 'videos/chunk_000/observation.images.vid_dir_wrist1'
        if not os.path.exists(self.vid_dir_wrist1):
            os.makedirs(self.vid_dir_wrist1)

        self.vid_dir_writst2 =SAVE_DIR + 'videos/chunk_000/observation.images.vid_dir_wrist2'
        if not os.path.exists(self.vid_dir_writst2):
            os.makedirs(self.vid_dir_writst2)

        self.vid_dir_head =SAVE_DIR + 'videos/chunk_000/observation.images.vid_dir_head'
        if not os.path.exists(self.vid_dir_head):
            os.makedirs(self.vid_dir_head)

        # episode index
        self.episode_index = 0
        self.index = 0

        self.reset()

    def reset(self):
        self.df = pd.DataFrame(columns=['observation.state', 'action', 'timestamp', 'episode_index', 'frame_index', 'index', 'next.reward', 'next.done', 'task_index'])
        
        self.timestamp = 0
        self.frame_index = 0

        self.wrist1_camera_array = []
        self.wrist2_camera_array = []
        self.head_camera_array = []

        # this is for data logging 
        self.column_index = 0

    def write_data_to_buffer(self, observation, action, reward, termination_flag, cam_data, debug_stuff):
        if debug_stuff[1] % (debug_stuff[0]//5) == 0:
            print(f"Write Data: {(debug_stuff[1]/debug_stuff[0]) * 100:.3f}%")
        #print(termination_flag)
        # save it into local memory 

        # https://docs.phospho.ai/learn/lerobot-dataset
        # LeRobot want their .parquet to have:
        # observation.state, action, timestamp, episode_index, frame_index, index, next.done(optional), task_index(optional)
        # we can also include next.reward and next.done it seems like
        self.df.loc[self.column_index] = [observation['policy'].cpu().numpy()[0], action.cpu().numpy()[0], self.timestamp, 
                                          self.episode_index, self.frame_index, self.index, reward.cpu().item(), 
                                          termination_flag.cpu().item(), 0]
        self.column_index += 1
        # TODO: Update timestamp
        self.frame_index += 1
        self.index += 1

        #print(self.df)
        # exit()

        # TODO: append camrea data

    def dump_buffer_data(self):
        print("Start Writing Data")
        # exit()
        # dump all the data into corect dir :(
        if self.episode_index <= 9:
            data_file_name = 'episode_00000' + str(self.episode_index) + '.parquet'
            video_file_name = 'episode_00000' + str(self.episode_index) + '.mp4'
        elif 9 < self.episode_index <= 99:
            data_file_name = 'episode_0000' + str(self.episode_index) + '.parquet'
            video_file_name = 'episode_0000' + str(self.episode_index) + '.mp4'
        elif 99 < self.episode_index <= 999:
            data_file_name = 'episode_000' + str(self.episode_index) + '.parquet'
            video_file_name = 'episode_000' + str(self.episode_index) + '.mp4'
        elif 999 < self.episode_index <= 9999:
            data_file_name = 'episode_00' + str(self.episode_index) + '.parquet'
            video_file_name = 'episode_00' + str(self.episode_index) + '.mp4'
        else:
            data_file_name = 'episode_0' + str(self.episode_index) + '.parquet'
            video_file_name = 'episode_0' + str(self.episode_index) + '.mp4'

        table = pa.Table.from_pandas(self.df)
        pq.write_table(table, self.log_dir + data_file_name)


        #TODO:dump video data

        self.episode_index += 1
        print(f"Complete Writing Data. Saved to {self.log_dir + data_file_name}")


    def format_data(self, ):
        pass