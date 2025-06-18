from huggingface_hub import HfApi
import os

# api= HfApi(token=os.getenv(""))
# api.upload_folder(
#     folder_path="/home/isaac/Documents/Github/IsaacLab/lerobotTest",
#     repo_id="ssangjunpark/lerobotTest",
#     repo_type="dataset",
# )



hub_api = HfApi()
hub_api.create_tag("ssangjunpark/daros18_1510", tag="v2.0", repo_type="dataset")