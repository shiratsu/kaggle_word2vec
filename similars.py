# -*- coding: utf-8 -*-

from gensim.models import word2vec
import sys

model   = word2vec.Word2Vec.load(sys.argv[1])
results = model.most_similar(positive='老人', topn=10)

for result in results:
    print(result[0], '\t', result[1])