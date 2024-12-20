'''
Easy Inference
All you need to do is fill this file with the inference module of your model.
If you want to make batch infer, please convert according to the logic of your model.

Dataset: 
'''

import os
import argparse
import json
from tqdm import tqdm
import shutil

import librosa
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
import torch

data_path_root = '/home/coder/AIR-Bench_Dataset/Chat'  #Chat dataset path
input_file = f'{data_path_root}/Chat_meta.json'
output_file = 'Chat_result_modelx.jsonl'

def main():
    #Step1: Build model
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-Audio-7B-Instruct", 
        device_map="cuda", 
        torch_dtype=torch.bfloat16
    )
    model.eval()

    #Step2: Single step inference
    with open(input_file, "r") as fin, open(output_file, "w") as fout:
        fin = json.load(fin)
        for item in tqdm(fin):
            wav = item['path']
            task_name = item['task_name']
            dataset_name = item['dataset_name']
            data_path = f'{data_path_root}/{task_name}_{dataset_name}/{wav}'
            if not os.path.exists(data_path):
                print(f"lack wav {data_path}")
                continue
            
            #Construct prompt
            question = item['question']
            instruction = question

            #Step 3: Run model inference
            # Inference를 위한 모델 코드를 아래에 입력
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio_url": data_path},
                    ],
                },
                # 여기서는 유저가 음성을 주고, 이어서 바로 텍스트 질문을 한다고 가정
                {
                    "role": "user",
                    "content": instruction
                }
            ]
            
            text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)

            audios = []
            for message_data in conversation:
                content = message_data["content"]
                if isinstance(content, list):
                    for ele in content:
                        if ele.get("type") == "audio":
                            y, sr = librosa.load(ele['audio_url'], 
                                                 #sr=processor.feature_extractor.sampling_rate
                                                )
                            audios.append(y)

            inputs = processor(text=text, audios=audios, return_tensors="pt", padding=True)
            
            # 모든 tensor를 GPU로 이동
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to("cuda")

            with torch.no_grad():
                generate_ids = model.generate(**inputs, max_length=256)
            
            # 인코딩 길이만큼 잘라서 실제 생성된 답변 부분만 추출
            generate_ids = generate_ids[:, inputs["input_ids"].size(1):]

            output = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

            #Step 4: save result
            json_string = json.dumps(
                {
                    "meta_info": item['meta_info'],
                    "question": question,
                    "answer_gt": item["answer_gt"],
                    "path": item["path"],
                    "task_name": task_name,
                    "dataset_name": dataset_name,
                    "response": output,
                    "uniq_id": item["uniq_id"],
                },
                ensure_ascii=False
            )
            fout.write(json_string + "\n")

if __name__ == "__main__":
    main()
