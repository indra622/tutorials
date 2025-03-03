# -*- coding: utf-8 -*-
import json
import os
import random
import time
import logging
import shortuuid
from copy import deepcopy
from concurrent.futures.thread import ThreadPoolExecutor
import threading
import argparse
from func_timeout import func_set_timeout
import requests
import jsonlines
import pandas as pd
from tqdm import tqdm
import openai

MAX_API_RETRY = 3
LLM_MIT_RETRY_SLEEP = 5

# 환경 변수 설정
os.environ['OPENAI_API_KEY'] = 'your openai key'

lock = threading.Lock()
finish_count = 0
failed_count = 0

import logging

# 로거 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d %(levelname)s:\t%(message)s",
    datefmt="%Y-%m-%d,%H:%M:%S",
)
logger = logging.getLogger(__name__)  # 로거 객체 생성


def load_file2list(path):
    res = []
    with open(path, 'r+', encoding='utf8') as f:
        for item in jsonlines.Reader(f):
            res.append(item)
    return res

def openai_api_call(**kwargs):
    """
    OpenAI API 호출 함수
    """
    openai.api_key = os.environ.get("OPENAI_API_KEY")

    for _ in range(MAX_API_RETRY):
        try:
            response = openai.chat.completions.create(**kwargs)
            return response
        except openai.OpenAIError as e:
            print(f"Error: {e}")
            time.sleep(LLM_MIT_RETRY_SLEEP)
    return None

def get_result_by_request(**kwargs):
    """
    OpenAI API 호출 후 결과 반환
    """
    response = openai_api_call(**kwargs)
    if response:
        choices = response.choices[0]
        result = choices.message.content
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        finish_reason = choices.finish_reason
        return result, prompt_tokens, completion_tokens, finish_reason

    else:
        raise Exception("Failed to get a valid response from OpenAI API")

def task(data, writer, args):
    global finish_count
    global failed_count
    openai_args = {
        'model': args.model_name,
        'temperature': args.temperature,
        'max_tokens': args.max_tokens,
        'messages': data['openai_args']['messages']
    }
    
    for _ in range(MAX_API_RETRY):
        try:
            result, prompt_tokens, completion_tokens, finish_reason = get_result_by_request(**openai_args)
            item = deepcopy(data)
            item.update({
                "gen": result,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "finish_reason": finish_reason
            })
            lock.acquire()
            finish_count += 1
            writer.write(json.dumps(item, ensure_ascii=False) + "\n")
            writer.flush()
            lock.release()
            return

        except Exception as e:
            print("request error", e)
            pass

    lock.acquire()
    failed_count += 1
    lock.release()

def get_unprocessed_data(args, out_file):
    data_l = list()
    uuid_s = set()
    if os.path.exists(out_file):
        out_data_l = load_file2list(out_file)
        for data in out_data_l:
            if data['gen'] != 'error':
                uuid_s.add(data[args.uuid])
    for data in load_file2list(args.in_file):
        if data[args.uuid] in uuid_s:
            continue
        data_l.append(data)
    return data_l

def run_chat_gen(args):
    if args.out_file:
        out_file = args.out_file
    else:
        out_file = os.path.splitext(args.in_file)[0] + '_result.jsonl'
    items = get_unprocessed_data(args, out_file)

    pool = ThreadPoolExecutor(max_workers=args.num_workers)
    writer = open(out_file, 'a', encoding='utf8')
    total_count = 0
    global finish_count, failed_count
    for item in items:
        total_count += 1
        pool.submit(task, item, writer, args)
    while finish_count + failed_count < total_count:
        logger.info(f"total:{total_count}  finish:{finish_count}  failed:{failed_count}")
        time.sleep(10)

    time.sleep(10)
    writer.close()

def build_test_file():
    root = '.'

    with open(os.path.join(root, 'batch_run_input.jsonl'), 'w', encoding='utf-8') as writer:
        with open('Chat_result_modelx.jsonl', 'r') as fp:
            for data in fp:
                row = json.loads(data)

                system_prompt = ("You are a helpful and precise assistant for checking the quality of the answer.\n"
                                 "[Detailed Audio Description]\nXAudioX\n[Question]\nXQuestionX\n"
                                 "[The Start of Assistant 1s Answer]\nXAssistant1X\n[The End of Assistant 1s Answer]\n"
                                 "[The Start of Assistant 2s Answer]\nXAssistant2X\n[The End of Assistant 2s Answer]\n[System]\n"
                                 "We would like to request your feedback on the performance of two AI assistants in response to the user question "
                                 "and audio description displayed above. AI assistants are provided with detailed audio descriptions and questions.\n"
                                 "Please rate the helpfulness, relevance, accuracy, and comprehensiveness of their responses. "
                                 "Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance. "
                                 "Please output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. "
                                 "The two scores are separated by a space.")

                path = row['path']
                question = row['question']
                answer_gt = row['answer_gt']
                task_name = row['task_name']
                dataset_name = row['dataset_name']
                response = row['response']

                if response is None:
                    continue

                if row.get('meta_info', None) is None:
                    print("lack meta info")
                    exit(1)
                else:
                    meta_info = row['meta_info']

                content = system_prompt.replace("XAudioX", meta_info).replace("XQuestionX", question).replace("XAssistant1X", answer_gt).replace("XAssistant2X", response)
                tmp_d = {
                    'uuid': shortuuid.uuid(),
                    'openai_args': {
                        "messages": [{"role": "user", "content": content}]
                    },
                    'meta_info': meta_info,
                    'path': path,
                    'question': question,
                    'answer_gt': answer_gt,
                    'task_name': task_name,
                    'dataset_name': dataset_name,
                    'Audio-LLM-response': response,
                }
                if random.random() < 0.5:
                    tmp_d['openai_args'].update({"temperature": 2.0})
                writer.write(json.dumps(tmp_d, ensure_ascii=False) + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="llm gen")
    parser.add_argument("-r", "--root", type=str)
    parser.add_argument("-i", "--in-file", type=str, default='batch_run_input.jsonl')
    parser.add_argument("-o", "--out-file", type=str, default='batch_run_output.jsonl')
    parser.add_argument("-n", "--num-workers", type=int, default=1)  # max=50
    parser.add_argument("-m", "--model-name", type=str, default='gpt-4')
    parser.add_argument("-t", "--temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--uuid", type=str, default='uuid')

    args = parser.parse_args()

    # Step 1
    build_test_file()

    # Step 2
    run_chat_gen(args)
