import sentencepiece as spm
import os
from nltk import sent_tokenize
import json
import pandas as pd
from sklearn.utils import shuffle
import langid
def file_to_df(input_path, classification, sent_count = 100000):
    all_docs = []
    counter = 0
    num_words = 0
    with open(input_path, 'r', encoding='utf8') as f:
        for line in f:
            counter += 1
            if counter % 10000 == 0:
                print('Processing json: ', counter)
            line = json.loads(line)
            title = line.get('title') or ''
            abstract = line.get('abstract') or ''

            text = title + '. ' + abstract
            num_words = num_words + len(text.split())
            kw = ""

            try:
                lang = line['lang']
            except:
                lang = langid.classify(text)[0]

            all_docs.append([text,kw,lang])

    df = pd.DataFrame(all_docs)

    df.columns = ["text", "keyword","lang"]
    if sent_count <= len(df):
        df = df.sample(sent_count, weights = df.groupby('lang')['lang'].transform('count')).reindex()

    print(input_path, 'data size: ', df.shape)
    print('Avg words: ', num_words/sent_count)
    return df

def train_bpe_model(input, output):
    if input is not None:
        df = file_to_df(os.path.join(input), classification=False)
        with open(output + '.txt', 'w', encoding='utf8') as f:
            for idx, line in df.iterrows():
                text = line['text']
                sents = sent_tokenize(text)
                for sent in sents:
                    f.write(sent.lower().strip() + '\n')
    
    assert not os.path.exists(output + '.model')
    spm.SentencePieceTrainer.Train('--input=' + output + '.txt --model_prefix=' + output + ' --vocab_size=32000 --character_coverage=1.0')


    sp = spm.SentencePieceProcessor()
    sp.Load(output + ".model")

    os.remove(output + '.txt')
"""
import itertools
if __name__ == '__main__':
    lang = ['croatian','estonian','latvian','russian']
    for k in range(1,2):
        for l in list(itertools.combinations(lang,k)):
            input =  "_".join(list(l)+['big.json'])
            print(input)
            #input = 'estonian_big.json'
            output = "../data/bpe_"+"_".join(list(l))+"_big"
            train_bpe_model(input, output)

"""