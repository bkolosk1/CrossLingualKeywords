import os
import json
from tqdm import tqdm
PATH_DATASETS = '.'


lang2idx = {
    'english': 0,
    'latvian': 1,
    'croatian': 2,
    'russian': 3,
    'estonian': 4,
    'slovenian': 5
}


for language in tqdm(os.listdir(PATH_DATASETS)):
    print(f"Currently processing: {language}")
    train_path = os.path.join(PATH_DATASETS,language,language+"_train.json")
    valid_path = os.path.join(PATH_DATASETS,language,language+"_valid.json")
    if os.path.exists(train_path):
        os.rename(train_path, valid_path)

for language in tqdm(os.listdir(PATH_DATASETS)):
    print(f"Currently processing: {language}")
    if language[-2:] == 'py': continue

    for split in ['_valid.json', '_test.json','_all.json']:
        split_path = os.path.join(PATH_DATASETS,language,language+split)
        split_output_open = os.path.join(PATH_DATASETS,language,language+"_tmp"+split)
        output_json = open(split_output_open, 'w') 
        with open(split_path, "r") as split_json:
            for line in split_json:
                current_line = json.loads(line)
                current_line['lang'] = language
                output_json.write(json.dumps(current_line, ensure_ascii=False)+"\n")
        output_json.close()

        #REMOVE OLD 
        os.remove(split_path)
        #INSERT NEW
        os.rename(split_output_open, split_path)
    