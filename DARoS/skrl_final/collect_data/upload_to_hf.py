from huggingface_hub import HfApi
import os

hub_api = HfApi()
hub_api.create_tag("ssangjunpark/daros29_0754", tag="v2.0", repo_type="dataset")