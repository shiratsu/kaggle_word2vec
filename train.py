# ライブラリ読み込み
from gensim.models import word2vec
import re, pprint
import logging
import sys

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

data = word2vec.Text8Corpus('/Users/shunsukehiratsuka/PycharmProjects/kaggleWord2vec/data/wakati_rojinto_umi.txt')
model = word2vec.Word2Vec(data,
                          sg=1,
                          size=100,
                          min_count=1,
                          window=10,
                          hs=1,
                          negative=0)
model.save(sys.argv[1])