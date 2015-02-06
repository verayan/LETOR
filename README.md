# Learning to rank

This project is about training ranking models so that top ranked documents relevant to queries can be returned.

The data set used is TREC dataset in LETOR from MSA. There are 44 features associated with each document including BM25, tf-idf and pagerank. More detailed info about the dataset can be found in Readme_TREC_dataset.pdf

To train pair-wise LETOR model, the classification model(logistic regression and SVM) are used to find the best parameter values. A linear combination of features weighted by these parameter values can be used to give scores for webpages based on their relevance.

