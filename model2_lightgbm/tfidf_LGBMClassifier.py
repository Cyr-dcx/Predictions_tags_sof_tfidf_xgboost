
import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, jaccard_score, hamming_loss,zero_one_loss, accuracy_score



df = pd.read_csv("../data/question_tags_df.csv")
vocab = df['Question'].tolist()

tfidf = TfidfVectorizer(analyzer="word",
                        min_df=0.004,
                        max_df=0.6,
                        lowercase=False,
                        ngram_range = (1, 2),
                        dtype=np.float32
                       )
tfidf.fit(vocab)

metrics_df = pd.DataFrame(columns=["embedding","model","f1","jaccard_score","hamming_loss","zero_one_loss"])

def add_a_row(df, embedding, model, f1, jaccard_score,hamming_loss,zero_one_loss):
    df = df.append({"embedding" :embedding , 
               "model" :model, 
               "f1":f1, 
               "jaccard_score":jaccard_score,
               "hamming_loss":hamming_loss,
               "zero_one_loss":zero_one_loss}, ignore_index=True)
    return df

X = df["Question"]
y = [set(str(element).split(",")) for element in df["tags"]]

from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()
y_multilabel_encoded = mlb.fit_transform(y)
tags = list(mlb.classes_)



X_train, X_test, y_train, y_test = train_test_split(X,y_multilabel_encoded,test_size=0.3)
X_tfidf = tfidf.transform(X)
X_train_vec = tfidf.transform(X_train)
X_test_vec = tfidf.transform(X_test)

def get_best_thresholds(y_test,y_pred):
  thresholds = [i/100 for i in range(100)]
  best_thresholds = []
  for idx in range(20):
    f1_scores = [f1_score(y_test[:, idx], (y_pred[:, idx] > thresh) * 1) for thresh in thresholds]
    best_thresh = thresholds[np.argmax(f1_scores)]
    best_thresholds.append(best_thresh)
  return best_thresholds

## one vs all classififier with light boost
onevsall_lgmb = OneVsRestClassifier(lgb.LGBMClassifier())
onevsall_lgmb.fit(X_train_vec, y_train)
y_proba_lgmb = onevsall_lgmb.predict(X_test_vec)

best_thresholds_lgbm = get_best_thresholds(y_test, y_proba_lgmb)

y_pred_lgbm = np.empty_like(y_proba_lgmb)
for i, thresh in enumerate(best_thresholds_lgbm):
  y_pred_lgbm[:, i] = (y_proba_lgmb[:, i] > thresh) * 1
  
# Evaluate the model
f1_score_lightGBM = f1_score(y_test, y_pred_lgbm, average = "micro")
jaccard_score_lightGBM = jaccard_score(y_test, y_pred_lgbm, average="micro")
hamming_loss_lightGBM = hamming_loss(y_test, y_pred_lgbm)
zero_one_loss_lightGBM = zero_one_loss(y_test, y_pred_lgbm, normalize=True)

metrics_df = add_a_row(metrics_df,"tfidf","OneVsrest-LGBMClassifier",f1_score_lightGBM,jaccard_score_lightGBM,hamming_loss_lightGBM,zero_one_loss_lightGBM)

filename = 'tfidf_lightGBM.sav'
joblib.dump(onevsall_lgmb,filename)

print(metrics_df)


