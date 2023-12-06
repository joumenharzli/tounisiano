#!/usr/bin/env python3

import requests
import json

def download_file(url, destination):
    response = requests.get(url)
    with open(destination, 'wb') as file:
        file.write(response.content)

def load_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

        # Initialize an empty list to store the result
        result_list = []

        # Iterate through the "data" key in the JSON structure
        for item in data["data"]:
            for paragraph in item["paragraphs"]:
                paragraph_list = []
                for qa in paragraph["qas"]:
                    # Extract question and answer information
                    question_text = qa["question"].replace("- +","").strip()
                    answer_text = qa["answers"][0]["text"].strip()  # Assuming there is only one answer

                    # Append question and answer to the result list
                    paragraph_list.append({"question": question_text, "answer": answer_text})
                result_list.append(paragraph_list)

    return result_list

def format_dataset(result_list):
    result_text_list = []
    for paragraph in result_list:
        paragraph_text = ""
        for qa in paragraph:
            paragraph_text += "<|im_start|>user "+qa["question"]+"<|im_end|>\n<|im_start|>assistant "+qa["answer"]+"<|im_end|>\n"
        paragraph_text = paragraph_text.rstrip()
        result_text_list.append(paragraph_text)

download_file("https://storage.googleapis.com/ext-oss-tunbert-gcp/TRCD_dataset/TRCD_train.json","TRCD_train.json")
download_file("https://storage.googleapis.com/ext-oss-tunbert-gcp/TRCD_dataset/TRCD_valid.json","TRCD_valid.json")
download_file("https://storage.googleapis.com/ext-oss-tunbert-gcp/TRCD_dataset/TRCD_test.json","TRCD_test.json")

