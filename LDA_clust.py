from __future__ import division, print_function
from time import time
import numpy as np
import sklearn.feature_extraction.text as text
import lda
import pandas as pd

t0 = time()

n_samples = 2000
n_features = 1000
n_topics = 10
n_top_words = 20

docs = pd.read_csv('./docs.csv')

titles = tuple(docs.values.tolist())
tests = docs.ix[:,1].values.tolist()

tf_vectorizer = text.CountVectorizer(max_df=0.95, min_df=2, max_features=n_features, stop_words='english')
tf = tf_vectorizer.fit_transform(tests)

vocab = tf_vectorizer.get_feature_names()


# -----------------------------tf-idf (for NMF)------------------------------
# tfidf_vectorizer = text.TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
# tfidf = tfidf_vectorizer.fit_transform(tests)
# vocab = tfidf_vectorizer.get_feature_names()
# ---------------------------------------------------------------------------


model = lda.LDA(n_topics=20, n_iter=500, random_state=1)
model.fit(tf)

topic_word = model.topic_word_
n = 5
topic_details = []
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n+1):-1]
    topic_details.append([i, ',\n'.join(topic_words)])

topicDF = pd.DataFrame(topic_details)
topicDF.columns = ['topic', 'words']
topicDF.to_csv('./LDA_topics.csv', index=False)

# -------------------------------Print on console-------------------------------------
# for i, topic_dist in enumerate(topic_word):
#     topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n+1):-1]
#     print('*Topic {}\n- {}'.format(i, ' / '.join(topic_words)))
# ------------------------------------------------------------------------------------

doc_topic = model.doc_topic_
document_topics = []
for n in range(40):
    topic_most_pr = doc_topic[n].argmax()
    document_topics.append([n, topic_most_pr, titles[n][:50][1], topic_word])

docDF = pd.DataFrame(document_topics).ix[:,:2]
docDF.columns = ['id', 'topic', 'docs']
print('\n\nPrinting Results\n\nDocument Topic Distribution:\n\n',str(docDF))
docDF.to_csv('./LDA_clusters.csv', index=False)

# ------------------------------Print on console-------------------------------------
# for n in range(40):
#     topic_most_pr = doc_topic[n].argmax()
#     print("doc: {} topic: {}\n{}...".format(n,
#                                             topic_most_pr,
#                                             titles[n][:50]))
# -----------------------------------------------------------------------------------

print("done in %0.3fs." % (time() - t0))