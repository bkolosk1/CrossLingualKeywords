import pandas as pd
import os
import json
from bpe import bpe

PATH_DATASETS = '../CL_KW/'

def create_valid_config(languages):
    data_lang_paths = "_".join(languages)

    dir_cl_path = os.path.join("data",data_lang_paths)
    print(dir_cl_path)
    os.makedirs(dir_cl_path, exist_ok = True )
        
    import glob
    files = glob.glob(f'{dir_cl_path}/*')
    for f in files:
        os.remove(f)
    
    for split in ['valid','test']:
        split_paths = [os.path.join(PATH_DATASETS,lang,lang+f"_{split}.json")  for lang in languages]
        paths_in = " ".join(split_paths)
        paths_out =  os.path.join(dir_cl_path,data_lang_paths+f"_{split}.json")
        os.system(f"cat {paths_in} >> {paths_out}")
        os.system(f"wc -l {paths_in} {paths_out}")
        os.system(f"shuf {paths_out} -o {paths_out}")
   
"""

     

languages = ['estonian','latvian','english','russian','slovenian']
#STEP 1 create cross_ling config
create_valid_config(languages)

#STEP 2 create cross_linhg BPE
data_lang_paths = "_".join(languages)
input = os.path.join("./data/",data_lang_paths,data_lang_paths+"_all.json")
output = f"./bpe/bpe_{data_lang_paths}_big"
bpe.train_bpe_model(input, output)
"""