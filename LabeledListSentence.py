# ライブラリ読み込み
import gensim, logging

class LabeledListSentence(object):
    def __init__(self, words_list, labels):
        self.words_list = words_list
        self.labels = labels

    def __iter__(self):
        for i, words in enumerate(self.words_list):
            # yield LabeledSentence(words, ['SENT_%s' % self.labels[i]])
            yield gensim.models.doc2vec.TaggedDocument(words, ['%s' % self.labels[i]])