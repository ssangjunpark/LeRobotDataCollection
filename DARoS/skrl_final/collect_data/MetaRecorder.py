import numpy as np
import pandas as pd
import os
import json

import io
from PIL import Image

from os import listdir
from os.path import isfile, join

import concurrent.futures
import multiprocessing as mp

from DataCollectionConstants import SAVE_DIRECTORY_PATH_META, JOINT_NAMES

SAVE_DIR = SAVE_DIRECTORY_PATH_META

class MetaRecorder:
    def __init__(self, data_folder_path):
        self.data_folder_path = data_folder_path
        self.task = "Grab the door handle"

        self.num_workers = mp.cpu_count()

        if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR)

    def _all_data_path_finder(self):
        all_files = []
        for sub in os.listdir(self.data_folder_path):
            sub_path = self.data_folder_path + '/' + sub
            #print(sub_path)
            #exit()
            for file in os.listdir(sub_path):
                f_path = sub_path + '/' + file
                # print(f_path)   
                all_files.append(f_path)
        
        all_files.sort()
        return all_files

    
    def generate_episodes_jsonl(self):
        all_files = self._all_data_path_finder()

        jsonl_data = []

        for file in all_files:
            df = pd.read_parquet(file)
            
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

        self._write_data(jsonl_data, SAVE_DIR+"episodes.jsonl")

    def generate_info_json(self):
        all_files = self._all_data_path_finder()
        total_episodes = len(all_files)
        total_frames = 0
        unique_task_indices = set()
        fps = 0.0
        total_tasks = 0
        sample_features = {}

        data_path = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
        video_path = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"

        # just take on df for features since everything is the same
        inference_df = pd.read_parquet(all_files[0])
        for col in inference_df.columns:
            col_pointer = inference_df[col]
            # print(col_sample)
            # print(type(col_sample))
            sample = col_pointer.iloc[0]
            # print(sample)
            # exit()
            if isinstance(sample, (np.ndarray, list, tuple)):
                arr = np.array(sample)
                data_type = str(arr.dtype)
                data_shape = [int(str(arr.shape).split(',')[0].split('(')[1])]
            else:
                data_type = str(inference_df[col].dtype)
                data_shape = [1]
            
            sample_features[col] = {
                'dtype' : data_type,
                'shape' : data_shape,
                'names' : JOINT_NAMES # I think it is easier to manually fill this out since getting the joint name via script will be a pain...
            }

            for key, val in sample_features.items():
                if key.startswith("observation.images."):
                    val['dtype'] = 'image'
                    val['shape'] = [256, 256, 3]
                    val['names'] = ["height", "width", "channel"]
        
        # print(sample_features) 
        # {'observation.images.image': {'dtype': 'object', 'shape': [1], 'names': 'TODO'}, 'observation.state': 
        # {'dtype': 'float32', 'shape': ['45'], 'names': 'TODO'}, 'action': {'dtype': 'float32', 'shape': ['13'], 'names': 'TODO'}, 
        # 'timestamp': {'dtype': 'float64', 'shape': [1], 'names': 'TODO'}, 'episode_index': {'dtype': 'int64', 'shape': [1], 'names': 'TODO'}, 
        # 'frame_index': {'dtype': 'int64', 'shape': [1], 'names': 'TODO'}, 'index': {'dtype': 'int64', 'shape': [1], 'names': 'TODO'}, 
        # 'next.reward': {'dtype': 'float64', 'shape': [1], 'names': 'TODO'}, 'next.done': {'dtype': 'bool', 'shape': [1], 'names': 'TODO'}, 
        # 'task_index': {'dtype': 'int64', 'shape': [1], 'names': 'TODO'}}
        # exit()

        fps_l = []
        for file in all_files:
            df = pd.read_parquet(file)
            total_frames += len(df)
            unique_task_indices.update(df['task_index'].unique().tolist())
            ts = df['timestamp'].values
            # print(type(ts))
            diffs = np.diff(ts) # type: ignore
            # print(diffs)
            # exit()
            fps_est = 1.0 / np.mean(diffs)
            fps_l.append(fps_est)

        fps = float(np.mean(fps_l))
        total_tasks = len(unique_task_indices)

        splits = {"train":f"0:{total_episodes}"}

        info = {
            "codebase_version": "v2.0",
            "robot_type": "realmandoor",
            "total_episodes": total_episodes,
            "total_frames": total_frames,
            "total_tasks": total_tasks,
            "total_videos": 0,
            "total_chunks": 1,
            "chunks_size": 1000,
            "fps": fps,
            "splits" : splits,
            "data_path" : data_path,
            "video_path" : video_path, 
            "features": sample_features
        }

        out_f_name = SAVE_DIR+'info.json'
        with open(out_f_name, 'w') as f:
            json.dump(info, f, indent=4)
    
        print(f"Successfully generated {out_f_name} to {os.getcwd()}")
        print("WARNING (info.json): \ninfo.json 1) Need to mannually write shapes for image data (REQUIRED)\ninfo.json 2) add joint/motor names (if needed)")

    def generate_stats_json(self):
        all_files = self._all_data_path_finder()

        preprocessed_stats = {}

        processed_file_count = 0

        for file in all_files:
            df = pd.read_parquet(file)

            for col in df.columns:
                #first create place holder with sample on top
                if col not in preprocessed_stats:
                    col_pointer = df[col]
                    sample = col_pointer.iloc[0]
                    # print(sample)
                    # exit()
                    # when we see image in bytes 
                    if isinstance(sample, dict) and 'bytes' in sample:
                        img = Image.open(io.BytesIO(sample['bytes']))
                        img = img.convert("RGB")
                        # i think we should normalize since hugging face sample do that - prob becasuse of how fed into NN by mdp.image
                        arr = np.asarray(img).astype(np.float32) / 255.0
                        C = arr.shape[2]
                        # create place holder
                        preprocessed_stats[col] = {
                            "dt" : "img",
                            "sum" : np.zeros((C,), dtype=np.float64),
                            "sum_sq": np.zeros((C,), dtype=np.float64),
                            "pixel_count" : 0,
                            "min" : np.full((C,), np.inf, dtype=np.float64),
                            "max" : np.full((C,), -np.inf, dtype=np.float64)
                        }
                    # when we see other numerical data
                    else:
                        if isinstance(sample, (list, tuple, np.ndarray)):
                            arr = np.asarray(sample, dtype=np.float64)
                        else:
                            arr = np.array([sample], dtype=np.float64)
                        #print(np.zeros(arr.flatten().shape, dtype=np.float64).shape)
                        #exit()
                        preprocessed_stats[col] = {
                            "dt" : "num",
                            "sum" : np.zeros(arr.flatten().shape, dtype=np.float64),
                            "sum_sq": np.zeros(arr.flatten().shape, dtype=np.float64),
                            "count" : 0,
                            "min" : np.full(arr.flatten().shape, np.inf, dtype=np.float64),
                            "max" : np.full(arr.flatten().shape, -np.inf, dtype=np.float64)
                        }
                
                col_dict_point = preprocessed_stats.get(col)
                #print(col_dict_point)
                
                df_rows = df[col]

                if col_dict_point['dt'] == "img": # type: ignore
                    for row in df_rows:
                        #print(row)
                        b = row.get('bytes', None) if isinstance(row, dict) else None
                        img = Image.open(io.BytesIO(b)) # type: ignore
                        img = img.convert("RGB")
                        arr = np.asarray(img).astype(np.float32) / 255.0

                        H, W, C = arr.shape
                        flat = arr.reshape(-1, C)
                        sum = flat.sum(axis=0)
                        sumsq = (flat * flat).sum(axis=0)
                        col_dict_point["sum"] += sum # type: ignore
                        col_dict_point["sum_sq"] += sumsq # type: ignore
                        col_dict_point["pixel_count"] += (H*W) # type: ignore

                        col_dict_point['min'] = np.minimum(flat.min(axis=0), col_dict_point["min"]) # type: ignore
                        col_dict_point['max'] = np.maximum(flat.max(axis=0), col_dict_point["max"]) # type: ignore

                        #print(col_dict_point)
                else:
                    for row in df_rows:
                        if isinstance(row, (list, tuple, np.ndarray)):
                            arr = np.asarray(row, dtype=np.float64).flatten()
                        else:
                            arr = np.array([row], dtype=np.float64)

                        col_dict_point["sum"] += arr # type: ignore
                        col_dict_point["sum_sq"] += arr * arr # type: ignore
                        col_dict_point["count"] += 1 # type: ignore

                        col_dict_point['min'] = np.minimum(arr, col_dict_point["min"]) # type: ignore
                        col_dict_point['max'] = np.maximum(arr, col_dict_point["max"]) # type: ignore
            
            processed_file_count += 1
            print(f"(stats.json): {processed_file_count}/{len(all_files)} processed ")

        dump_dict = {}
        # print(preprocessed_stats)
        
        for key, value in preprocessed_stats.items():
            if value['dt'] == 'img':
                pix_count = value["pixel_count"]
                mean = (value["sum"] / pix_count).tolist()
                var = (value["sum_sq"] / pix_count) - np.square(value["sum"] / pix_count)
                var = np.clip(var, a_min=0.0, a_max=None)
                min = value["min"].tolist()
                max = value["max"].tolist()

                dump_dict[key] = {
                    "mean" : self._format_for_RGB(mean),
                    "std" : self._format_for_RGB(np.sqrt(var).tolist()),
                    "min" : self._format_for_RGB(min),
                    "max" : self._format_for_RGB(max),
                }
                # print(dump_dict)
                
            else:
                count = value['count']
                mean = (value["sum"] / count).tolist()
                var = (value["sum_sq"] / count) - np.square(value["sum"] / count)
                var = np.clip(var, a_min=0.0, a_max=None)
                min = value["min"].tolist()
                max = value["max"].tolist()

                dump_dict[key] = {
                    "mean" : mean,
                    "std" : np.sqrt(var).tolist(),
                    "min" : min,
                    "max" : max,
                }
        
        #out_f_name = 'episodes_stats.jsonl'
        #with open(out_f_name, 'w') as f:
        #    f.write(json.dumps(dump_dict))
        #    f.write('\n')

        out_f_name = SAVE_DIR+'stats.json'
        with open(out_f_name, 'w') as f:
            json.dump(dump_dict, f, indent=4)
        print("WARNING (stats.jsonl): \nstats.jsonl 1) Need to mannually convert bool min max (REQUIRED)")

    def generate_stats_json_multiprocessing(self):
        all_files = self._all_data_path_finder()

        paths = all_files
        #print(paths)

        # I think this is a very good tutorial: https://www.youtube.com/watch?v=fKl2JW_qrso
        all_stats = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            ff = [executor.submit(_process_parquet_worker, path) for path in paths]

            counter = 0
            for f in concurrent.futures.as_completed(ff):
                l_stats = f.result()
                all_stats.append(l_stats)
                counter += 1
                print(f"(stats.json): {counter}/{len(all_files)} processed")

        
        preprocessed_stats = {}
        # print(all_stats)
        for stat in all_stats:
            for key, val in stat.items():
                if key not in preprocessed_stats:
                    preprocessed_stats[key] = val.copy()
                else:
                    if val['dt'] =='img':
                        preprocessed_stats[key]["sum"] += val['sum']
                        preprocessed_stats[key]["sum_sq"] += val['sum_sq']
                        preprocessed_stats[key]["pixel_count"] += val['pixel_count']
                        preprocessed_stats[key]["min"] = np.minimum(val['min'], preprocessed_stats[key]["min"])
                        preprocessed_stats[key]["max"] = np.maximum(val['max'], preprocessed_stats[key]["max"])
                    else:
                        preprocessed_stats[key]["sum"] += val['sum']
                        preprocessed_stats[key]["sum_sq"] += val['sum_sq']
                        preprocessed_stats[key]["count"] += val['count']
                        preprocessed_stats[key]["min"] = np.minimum(val['min'], preprocessed_stats[key]["min"])
                        preprocessed_stats[key]["max"] = np.maximum(val['max'], preprocessed_stats[key]["max"])

        dump_dict = {}

        for key, value in preprocessed_stats.items():
            if value['dt'] == 'img':
                pix_count = value["pixel_count"]
                mean = (value["sum"] / pix_count).tolist()
                var = (value["sum_sq"] / pix_count) - np.square(value["sum"] / pix_count)
                var = np.clip(var, a_min=0.0, a_max=None)
                min = value["min"].tolist()
                max = value["max"].tolist()

                dump_dict[key] = {
                    "mean" : self._format_for_RGB(mean),
                    "std" : self._format_for_RGB(np.sqrt(var).tolist()),
                    "min" : self._format_for_RGB(min),
                    "max" : self._format_for_RGB(max),
                }
                # print(dump_dict)
                
            else:
                count = value['count']
                mean = (value["sum"] / count).tolist()
                var = (value["sum_sq"] / count) - np.square(value["sum"] / count)
                var = np.clip(var, a_min=0.0, a_max=None)
                min = value["min"].tolist()
                max = value["max"].tolist()

                dump_dict[key] = {
                    "mean" : mean,
                    "std" : np.sqrt(var).tolist(),
                    "min" : min,
                    "max" : max,
                }

        out_f_name = SAVE_DIR+'stats.json'
        with open(out_f_name, 'w') as f:
            json.dump(dump_dict, f, indent=4)
        
        print("WARNING (stats.jsonl): \nstats.jsonl 1) Need to mannually convert bool min max (REQUIRED)")
        
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

        self._write_data(jsonl_data, SAVE_DIR+'tasks.jsonl')

    def _write_data(self, data, f_name):
        with open(f_name, 'w') as f:
            for l in data:
                f.writelines([json.dumps(l)])
                f.writelines('\n')

        print(f"Successfully generated {f_name} to {os.getcwd()}")

    def _format_for_RGB(self, lst):
        return [[[v]] for v in lst]

def _process_parquet_worker(path):
    # do waht ever did before but no more seen before thing 
    df = pd.read_parquet(path)
    stats = {}

    for col in df.columns:
        col_pointer = df[col]
        sample = col_pointer.iloc[0]

        if isinstance(sample, dict) and 'bytes' in sample:
            img = Image.open(io.BytesIO(sample['bytes']))
            img = img.convert("RGB")
            arr = np.asarray(img).astype(np.float32) / 255.0
            C = arr.shape[2]

            stats[col] = {
                "dt" : "img",
                "sum" : np.zeros((C,), dtype=np.float64),
                "sum_sq": np.zeros((C,), dtype=np.float64),
                "pixel_count" : 0,
                "min" : np.full((C,), np.inf, dtype=np.float64),
                "max" : np.full((C,), -np.inf, dtype=np.float64)
            }

            #unlike before now process that file here
            rows = df[col]
            for row in rows:
                b = row.get('bytes', None) if isinstance(row, dict) else None
                img = Image.open(io.BytesIO(b)) # type: ignore
                img = img.convert("RGB")
                arr = np.asarray(img).astype(np.float32) / 255.0

                H, W, C = arr.shape
                flat = arr.reshape(-1, C)
                sum = flat.sum(axis=0)
                sumsq = (flat * flat).sum(axis=0)
                stats[col]["sum"] += sum # type: ignore
                stats[col]["sum_sq"] += sumsq # type: ignore
                stats[col]["pixel_count"] += (H*W) # type: ignore

                stats[col]['min'] = np.minimum(flat.min(axis=0), stats[col]["min"]) # type: ignore
                stats[col]['max'] = np.maximum(flat.max(axis=0), stats[col]["max"]) # type: ignore

        else:
            if isinstance(sample, (list, tuple, np.ndarray)):
                arr = np.asarray(sample, dtype=np.float64)
            else:
                arr = np.array([sample], dtype=np.float64)
        
            stats[col] = {
                "dt" : "num",
                "sum" : np.zeros(arr.flatten().shape, dtype=np.float64),
                "sum_sq": np.zeros(arr.flatten().shape, dtype=np.float64),
                "count" : 0,
                "min" : np.full(arr.flatten().shape, np.inf, dtype=np.float64),
                "max" : np.full(arr.flatten().shape, -np.inf, dtype=np.float64)
            }

            #unlike before now process that file here
            rows = df[col]

            for row in rows:
                if isinstance(row, (list, tuple, np.ndarray)):
                    arr = np.asarray(row, dtype=np.float64).flatten()
                else:
                    arr = np.array([row], dtype=np.float64)

                stats[col]["sum"] += arr # type: ignore
                stats[col]["sum_sq"] += arr * arr # type: ignore
                stats[col]["count"] += 1 # type: ignore

                stats[col]['min'] = np.minimum(arr, stats[col]["min"]) # type: ignore
                stats[col]['max'] = np.maximum(arr, stats[col]["max"]) # type: ignore

    return stats