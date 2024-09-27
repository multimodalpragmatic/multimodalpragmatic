import easyocr
import os
import argparse
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch
import random
import numpy as np
import json



def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


# path_processor = {
#     'dalle3': "./results/dalle3",
#     'SD': "./results/SD",
#     'SDXL': "./results/SDXL",
#     'deepfloyd': "./results/deepfloyd",
#     'safe SD': "./results/safe SD",
#     'glyphcontrol': "/home/wiss/liu/Lucas/GlyphControl-release/imgs"
#
# }

import string


class OCR_Counter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.ocr_acc_list = []
        self.ocr_sub_correct_list = []
        self.ocr_sub_num_list = []

    def normalize_strings(self, words):
        # Remove punctuation using str.translate and string.punctuation
        table = str.maketrans('', '', string.punctuation)
        return [word.translate(table) for word in words]

    def update(self, text, ocr_result):
        ocr_text_list = [item[1] for item in ocr_result]
        ocr_text = ' '.join(ocr_text_list)
        text = text.lower()
        ocr_text = ocr_text.lower()

        if text[-1] == " ":
            text = text[:-1]
        if text[-1] == "!":
            text = text[:-1]
        if ocr_text[-1] == "!":
            ocr_text = ocr_text[:-1]
        res = text == ocr_text
        self.ocr_acc_list.append(res)

        words1 = ocr_text.split()
        # remove punctuations
        normalized_words1 = self.normalize_strings(words1)
        #         print(normalized_words1)
        words2 = text.split()
        match_count = sum(1 for word in words2 if word in normalized_words1)
        # use the number of match with the orignal text
        total_words = len(words2)
        #         sub_acc = matches/total_words
        self.ocr_sub_correct_list.append(match_count)
        self.ocr_sub_num_list.append(total_words)

    def output(self):
        self.acc = sum(self.ocr_acc_list) / len(self.ocr_acc_list)
        print("Final acc: ", self.acc)
        self.sub_acc = sum(self.ocr_sub_correct_list) / sum(self.ocr_sub_num_list)
        print("Sub string acc: ", self.sub_acc)


class JSONEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(JSONEncoder, self).default(obj)


def main(args):

    model_name = args.model_name
    prompt_mode = args.prompt_mode
    dataclass = args.dataclass

    # read data
    jsonl_file_path = './data/{}.jsonl'.format(dataclass)
    data = []
    with open(jsonl_file_path, 'r') as file:
        for line in file:
            # Parse the JSON object from each line and append to the list
            try:
                data.append(json.loads(line))
            except:
                print("Error in data")


    print(model_name)
    image_path = "./results/{}_{}_{}".format(model_name, prompt_mode, dataclass)

    ocr_result_dic = {}
    reader = easyocr.Reader(['en'], gpu=True)
    ocr_counter = OCR_Counter()


    for i, _data in tqdm(enumerate(data)):

        try:
            _image_path = image_path + "/{}.png".format(i)

            if model_name == "glyphcontrol":
                _image_path = image_path + "/{}_0.jpg".format(i)
            else:
                print("name is: ", model_name)

            print(_image_path)

            image = Image.open(_image_path)
            image_array = np.array(image)
            ocr_result = reader.readtext(image_array)
            ocr_result_dic[i] = ocr_result

            _text = _data["text_prompt"]
            ocr_counter.update(_text, ocr_result)
            print(i, ocr_result)
        except:
            print("Image{} does not exit".format(i))




    with open('./results/ocr/{}_{}_{}.json'.format(model_name, prompt_mode, dataclass), 'w') as json_file:
        json.dump(ocr_result_dic, json_file, cls=JSONEncoder)

    # acc
    ocr_counter.output()

    # save acc
    res = str(ocr_counter.acc)
    res2 = str(ocr_counter.sub_acc)
    file_path = './results/ocr/{}_{}_{}.txt'.format(model_name, prompt_mode, dataclass)
    if os.path.exists(file_path):
        # Delete the file
        os.remove(file_path)

    with open(file_path, 'a') as file:
        print(res, file=file)
        print("\n", file=file)
        print(res2, file=file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-model_name", type=str, default="safe SD")
    parser.add_argument("-prompt_mode", type=str, default="sign")
    parser.add_argument("-dataclass", type=str, default="hatespeech")

    args = parser.parse_args()

    set_seed(424242)

    main(args)