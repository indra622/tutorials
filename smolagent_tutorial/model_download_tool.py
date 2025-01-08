
import os
from dotenv import load_dotenv

load_dotenv()  # .env 파일에서 환경 변수를 읽어와 시스템 환경 변수에 추가

hf_token = os.getenv("HF_TOKEN")

import huggingface_hub
huggingface_hub.login(token=hf_token)
your_username='kresnik'

from smolagents import Tool

class HFModelDownloadsTool(Tool):
    name = "model_download_counter"
    description = """
    This is a tool that returns the most downloaded model of a given task on the Hugging Face Hub.
    It returns the name of the checkpoint."""
    inputs = {
        "task": {
            "type": "string",
            "description": "the task category (such as text-classification, depth-estimation, etc)",
        }
    }
    output_type = "string"

    def forward(self, task: str):
        from huggingface_hub import list_models

        model = next(iter(list_models(filter=task, sort="downloads", direction=-1)))
        return model.id

model_downloads_tool = HFModelDownloadsTool()

model_downloads_tool.push_to_hub(f"{your_username}/hf-model-downloads", token=f"{hf_token}")

