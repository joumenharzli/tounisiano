import json
from tounsiano.utils import utils


DATASETS_URLS = [
    "https://storage.googleapis.com/ext-oss-tunbert-gcp/TRCD_dataset/TRCD_train.json",
    "https://storage.googleapis.com/ext-oss-tunbert-gcp/TRCD_dataset/TRCD_valid.json",
    "https://storage.googleapis.com/ext-oss-tunbert-gcp/TRCD_dataset/TRCD_test.json",
]


def load_dataset(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

        # Initialize an empty list to store the result
        result_list = []

        # Iterate through the "data" key in the JSON structure
        for item in data["data"]:
            for paragraph in item["paragraphs"]:
                paragraph_list = []
                for qa in paragraph["qas"]:
                    # Extract question and answer information
                    question_text = qa["question"].replace("- +", "").strip()
                    answer_text = qa["answers"][0][
                        "text"
                    ].strip()  # Assuming there is only one answer

                    # Append question and answer to the result list
                    paragraph_list.append(
                        {"question": question_text, "answer": answer_text}
                    )
                result_list.append(paragraph_list)

    return result_list


# TODO: move to utils
def format_dataset(result_list):
    result_text_list = []
    for paragraph in result_list:
        paragraph_text = ""
        for qa in paragraph:
            paragraph_text += (
                "<|im_start|>user "
                + qa["question"]
                + "<|im_end|>\n<|im_start|>assistant "
                + qa["answer"]
                + "<|im_end|>\n"
            )
        paragraph_text = paragraph_text.rstrip()
        result_text_list.append(paragraph_text)


# TODO: save as json
def process_dataset():
    datasets = []
    for dataset_url in DATASETS_URLS:
        datasets.append(load_dataset(utils.download_file(dataset_url)))
