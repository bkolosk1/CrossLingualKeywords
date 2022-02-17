import string
import pke
from transformers.file_utils import tf_required
from prep_crossling import create_valid_config
from bert_crossling_prep import file_to_df
import multiprocessing as mp
import numpy as np
import stopwordsiso as stopwords
import string
from keybert import KeyBERT
from mrakun import RakunDetector




kw_model = KeyBERT()

from tqdm import tqdm
def solve_keybert(ps):
    df, stops = ps
    outs = []
    for i, row in tqdm(df.iterrows(),total=len(df)):
        text = row['text']
        lang = row['lang']
        curr_kw = []
        keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 3))#, diversity=0.7)
        print(keywords)
        #print(keywords)
        for k,prob in keywords:
            curr_kw.append(k)
        outs.append(curr_kw[:10])
    return outs

def solve_tfidf(ps):
    df, stops = ps
    outs = []
    for i, row in tqdm(df.iterrows(),total=len(df)):
        text = row['text']
        curr_kw = []
        extractor = pke.unsupervised.TfIdf()
        extractor.load_document(input=text)
        extractor.candidate_selection(n=3, stoplist=stops)
        extractor.candidate_weighting()
        keyphrases = extractor.get_n_best(n=10)
        for k,prob in keyphrases:
            curr_kw.append(k)
        outs.append(curr_kw[:10])
    return outs


def solve_MPR(ps):
    df, stops = ps
    outs = []
    for i, row in tqdm(df.iterrows(),total=len(df)):
        text = row['text']
        lang = row['lang']
        curr_kw = []
        extractor = pke.unsupervised.MultipartiteRank()
        extractor.load_document(input=text,
                                language=langdict[lang],
                                normalization=None)

        extractor.candidate_selection(stoplist=stops)
        extractor.candidate_weighting()
        keyphrases = extractor.get_n_best(n=10)
        for k,prob in keyphrases:
            curr_kw.append(k)
        outs.append(curr_kw[:10])
    return outs

def solve_yake(ps):
    df, stops = ps
    outs = []
    for i, row in tqdm(df.iterrows(),total=len(df)):
        text = row['text']
        lang = row['lang']
        curr_kw = []

        extractor = pke.unsupervised.YAKE()
        extractor.load_document(input=text,
                                language=langdict[lang],
                                normalization=None)
        extractor.candidate_selection(n=3, stoplist=stops)
        window = 2
        use_stems = False
        extractor.candidate_weighting(window=window,
                                    stoplist=stops,
                                    use_stems=use_stems) 
        threshold = 0.8
        keyphrases = extractor.get_n_best(n=10, threshold=threshold)

        for k,prob in keyphrases:
            curr_kw.append(k)
        outs.append(curr_kw[:10])
    return outs

def solve_kpminer(ps):
    df, stops = ps
    outs = []    
    for i, row in tqdm(df.iterrows(),total=len(df)):
        text = row['text']
        lang = row['lang']
        curr_kw = []
        extractor = pke.unsupervised.KPMiner()

        extractor.load_document(input=text,
                            language=langdict[lang],
                            normalization=None)

        lasf = 5
        cutoff = 200
        extractor.candidate_selection(lasf=lasf, cutoff=cutoff)
        extractor.candidate_weighting()
        keyphrases = extractor.get_n_best(n=10)
        for k,prob in keyphrases:
            curr_kw.append(k)
        outs.append(curr_kw[:10])
    return outs

def solve_topicalpagerank(ps):
    df, stops = ps
    outs = []    
    for i, row in tqdm(df.iterrows(),total=len(df)):
        text = row['text']
        lang = row['lang']
        curr_kw = []
        pos = {'NOUN', 'PROPN', 'ADJ'}
        grammar = "NP: {<ADJ>*<NOUN|PROPN>+}"
        extractor = pke.unsupervised.TopicalPageRank()
        extractor.load_document(input=text,
                                language=langdict[lang],
                                normalization=None)
        extractor.candidate_selection(grammar=grammar)
        extractor.candidate_weighting(window=10,
                                    pos=pos)
        keyphrases = extractor.get_n_best(n=10)
        for k,prob in keyphrases:
            curr_kw.append(k)
        outs.append(curr_kw[:10])
    return outs

def solve_textrank(ps):
    df, stops = ps
    outs = []    
    for i, row in tqdm(df.iterrows(),total=len(df)):
        text = row['text']
        lang = row['lang']
        curr_kw = []
        pos = {'NOUN', 'PROPN', 'ADJ'}
        extractor = pke.unsupervised.TextRank()
        extractor.load_document(input=text,
                                language=langdict[lang],
                                normalization=None)
        extractor.candidate_weighting(window=2,
                                    pos=pos,
                                    top_percent=0.33)
        keyphrases = extractor.get_n_best(n=10)
        for k,prob in keyphrases:
            curr_kw.append(k)
        outs.append(curr_kw[:10])
    return outs

def rakun_es(ps):
    df, stops = ps
    hyperparameters = {"distance_threshold":2,
                   "distance_method": "editdistance",
                   "num_keywords" : 10,
                   "pair_diff_length":2,
                   "stopwords" : stops,
                   "bigram_count_threshold":2,
                   "num_tokens":[1,2,3],
		            "max_similar" : 3, 
		            "max_occurrence" : 3} ## maximum frequency overall

    keyword_detector = RakunDetector(hyperparameters)
    outs = []
    for i, row in tqdm(df.iterrows(),total=len(df)):
        text = row['text']
        curr_kw = []
        keywords = keyword_detector.find_keywords(text, input_type = "text")
        for k,_ in keywords:
            curr_kw.append(k)
        outs.append(curr_kw[:10])
    return outs

def rakun_ft(ps):
    ft_to_lang = {
        "latvian" :  "./ft/wiki.lv.bin",
        "estonian" : "./ft/wiki.et.bin",
        "russian" : "./ft/wiki.ru.bin",
        "slovenian" : "./ft/wiki.sl.bin",
        "croatian" : "./ft/wiki.hr.bin",
        "english" : "./ft/wiki.en.bin"
    }
    df, stops = ps
    outs = []
    lang = df['lang'].to_list()[0]
    hyperparameters = {
                    "distance_threshold":3,
                   "num_keywords" : 10,
                   "distance_method": "fasttext",
                   "pair_diff_length":2,
                   "stopwords" :stops,
                   "pretrained_embedding_path": ft_to_lang[lang], 
                    "bigram_count_threshold":2,
                    "num_tokens":[1,2,3]}
    keyword_detector = RakunDetector(hyperparameters)   
    for i, row in tqdm(df.iterrows(),total=len(df)):
        text = row['text']
        #lang = row['lang']
     
        curr_kw = []
        keywords = keyword_detector.find_keywords(text, input_type = "text")
        for k,_ in keywords:
            curr_kw.append(k)
        outs.append(curr_kw[:10])
    return outs


langdict = {"russian":'ru',
            'latvian':'lv',
            'estonian':'ee',
            'slovenian':'sl',
            'croatian':'hr',
            'english':'en'}



WORKERS = 126

languages = ['english','russian','latvian','estonian','slovenian','croatian']
make_pairs = languages
print("ALL PAIRS: ",make_pairs)


for i,pair in enumerate(make_pairs):       

    stops = list(stopwords.stopwords(langdict[pair]))
    print(stops[:5])

    language_path = "_".join(pair)

    create_valid_config([pair])
    test_path = f"data/{pair}/{pair}_test.json"
    df_test = file_to_df(test_path,False,False)
    for method, method_f in [('KeyBERT',solve_keybert)], ('KPMiner', solve_kpminer), ('TextRank', solve_textrank), ('rakun-es', rakun_es),('yake', solve_yake), ('MPR', solve_MPR), ]:        
        out_k = []
        train_path = f"data/{pair}/{pair}_valid.json"
                            
        with mp.Pool(WORKERS) as pool:
            results = pool.map(method_f, [(df_t, stops) for df_t in np.array_split(df_test,WORKERS)])
        
        for r in results:
            out_k = out_k + r        

        outs = [k.split(';') for k in df_test['keyword']]
        print(out_k[:3], outs[:3])
        import pickle 
        with open(f'{method}_{pair}.pkl','wb') as f:
            pickle.dump(zip(out_k,outs), f)

