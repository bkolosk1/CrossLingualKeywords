# Data

* **Croatian, Estonian, Latvian and Russian** data, use this repository: https://www.clarin.si/repository/xmlui/handle/11356/1403
* **English** data, use this repository: https://github.com/ygorg/KPTimes
* **Slovenian** data, TO BE ADDED

Extract each language and prepare the data in a folder per language:  
To generate the splits, execute the following code:
```
python prepare_langs.py
python assert_working.py
```


# Citation

Croatian, Estonian, Latvian and Russian
```
 @misc{11356/1403,
 title = {Keyword extraction datasets for Croatian, Estonian, Latvian and Russian 1.0},
 author = {Koloski, Boshko and Pollak, Senja and {\v S}krlj, Bla{\v z} and Martinc, Matej},
 url = {http://hdl.handle.net/11356/1403},
 note = {Slovenian language resource repository {CLARIN}.{SI}},
 copyright = {Creative Commons - Attribution-{NonCommercial}-{NoDerivatives} 4.0 International ({CC} {BY}-{NC}-{ND} 4.0)},
 year = {2021} }
```

English
```
@inproceedings{gallina2019kptimes,
  title={KPTimes: A Large-Scale Dataset for Keyphrase Generation on News Documents},
  author={Gallina, Ygor and Boudin, Florian and Daille, B{\'e}atrice},
  booktitle={Proceedings of the 12th International Conference on Natural Language Generation},
  pages={130--135},
  year={2019}
}
```

Slovenian

```
```