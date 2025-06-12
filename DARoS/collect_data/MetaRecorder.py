import numpy as np
import pandas as pd
import os
import json

from os import listdir
from os.path import isfile, join

class MetaRecorder:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.task = "Grab the door handle"
    
    def generate_episodes_jsonl(self):
        all_files = [f for f in listdir(self.folder_path) if isfile(join(self.folder_path, f))]
        all_files.sort()

        jsonl_data = []

        for file in all_files:
            df = pd.read_parquet(self.folder_path + '/' + file)
            
            dump_dict = {}

            # print(type(df['episode_index'][0]))
            # print(df['episode_index'][0])
            # print(type(df['episode_index']))
            # exit()

            dump_dict['episode_index'] = int(df['episode_index'][0])
            dump_dict['tasks'] = [self.task]
            dump_dict['length'] = len(df)

            # print(dump_dict)
            # exit()

            jsonl_data.append(dump_dict)

        self._write_data(jsonl_data, "episodes.jsonl")

    def generate_info_json(self):
        print("need to mannually fill out info json for now... since it requires us to specify joint names, etc...")
        
    def generate_stats_json(self):
        pass

    def generate_tasks_jsonl(self, tasks):
        jsonl_data = []

        if tasks is None:
            tasks = [self.task]

        for task in tasks:
            dump_dict = {}
            
            dump_dict["task_index"] = 0
            dump_dict["task"] = task

            # print(dump_dict)
            # exit()
            
            jsonl_data.append(dump_dict)

        self._write_data(jsonl_data, 'tasks.jsonl')


    def _write_data(self, data, f_name):
        with open(f_name, 'w') as f:
            for l in data:
                f.writelines([json.dumps(l)])
                f.writelines('\n')

        print(f"Successfully generated {f_name} to {os.getcwd()}")

