# -*- coding: utf-8 -*-
# ライブラリ読み込み
from gensim import models
import LabeledListSentence


aryWebServer = [u'apache',u'nginx',u'tomcat',u'httpd']

aryWebPrograms = [u'php'
                ,u'ruby',u'python',u'java'
                ,u'laravel',u'fuelphp',u'symfony',u'fuelphp',u'phalcon'
                ,u'cakephp',u'script'
                ,u'seasor',u'struts',u'rails',u'django'
                ,u'swift'
               ]
aryJavascript = [u'bookmarklet',u'javascript',u'jquery',u'angularjs',u'backbone',u'nodejs']


aryInfra = [u'centos',u'ubuntu',u'aws',u'security',u'network',u'firewall'
            ,u'docker',u'vagrant',u'virtual',u'ec2',u's3',u'storage',u'ストレージ'
            ,u'ssh',u'iptables'
            ]

aryDB = [u'mysql',u'oracle',u'postgresql',u'db2',u'nosql',u'redis',u'mongodb']

aryFullText = [u'CloudSearch',u'Elastice Search',u'Apache solr',u'mysql']

aryCRM = [u'kintone',u'salesforce']

aryCI = [u'jenkins',u'travisci',u'circleci']

aryML = [u'chainer',u'tensorflow',u'keras',u'scikit-learn']

aryLabel = [u'webserver',u'programing language',u'javascript',u'infra',u'db',u'fulltext',u'crm',u'CI',u'機械学習']
aryPG = [aryWebServer,aryWebPrograms,aryJavascript,aryInfra,aryDB,aryFullText,aryCRM,aryCI,aryML]


print(aryPG)

# gensim にクチコミを登録
# クチコミに会社名を付与するため、参考記事で実装されていた拡張クラスを使っています
sentences = LabeledListSentence.LabeledListSentence(aryPG, aryLabel)

# doc2vec の学習条件設定
# alpha: 学習率 / min_count: X回未満しか出てこない単語は無視
# size: ベクトルの次元数 / iter: 反復回数 / workers: 並列実行数
model = models.Doc2Vec(alpha=0.025, min_count=1,
                       size=100, iter=20, workers=4)

# doc2vec の学習前準備(単語リスト構築)
model.build_vocab(sentences)

# Wikipedia から学習させた単語ベクトルを無理やり適用して利用することも出来ます
# model.intersect_word2vec_format('./data/wiki/wiki2vec.bin', binary=True)

# 学習実行
model.train(sentences)

# セーブ
model.save('./data/doc2vec.model')

# 学習後はモデルをファイルからロード可能
# model = models.Doc2Vec.load('./data/doc2vec.model')

# 順番が変わってしまうことがあるので会社リストは学習後に再呼び出し
checkLabel = model.docvecs.offset2doctag


results = model.most_similar(positive='php', topn=10)

for result in results:
    print(result[0], '\t', result[1])

print(checkLabel)