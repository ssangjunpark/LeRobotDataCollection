from MetaRecorder import MetaRecorder

from DataCollectionConstants import DATA_FOLDER_PATH

def main():
    meta_recorder = MetaRecorder(data_folder_path=DATA_FOLDER_PATH)
    # meta_recorder = MetaRecorder(data_folder_path="/home/sangjun-park/Desktop/LeRobotDataCollection/LeRobotData_18_2_9/data/chunk_000")

    meta_recorder.generate_episodes_jsonl()
    meta_recorder.generate_info_json()
    # meta_recorder.generate_stats_json()
    meta_recorder.generate_stats_json_multiprocessing()
    # it is bad desgin but non means hard coded task lol
    meta_recorder.generate_tasks_jsonl(tasks=None)

if __name__ == "__main__":
    main()