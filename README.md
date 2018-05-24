# DeepNLP Project

Study project on MIPT Deep NLP course.

Theme: Prediction of the Decision of the Arbitration Appellate Court of Russian Fededration in the Bankruptcy Case using Deep Learning.

The main goal of this study project was to collect judgements texts from public domain, make the
dataset and build deep learning model, that can predict from raw text. The task was formulated as binary
classification task.

Dataset: texts of 5500 bankruptcy cases of courts' judgements collected from web-site http://kad.arbitr.ru
4500 cases to train
1000 cases to test

Model used: Word2vec + LSTM

F1-score on test: 0.61


