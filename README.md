# Learning to rank

This project is about training ranking models so that top ranked documents relevant to queries can be returned.

The data set used is TREC dataset in LETOR from MSA. There are 44 features associated with each document including BM25, tf-idf and pagerank. More detailed info about the dataset can be found in Readme_TREC_dataset.pdf

To train pair-wise LETOR model, the classification model(logistic regression and SVM) are used to find the best parameter values. A linear combination of features are then weighted by these parameter values to give scores for webpages based on their relevance. The logistic regression model is implemented using stochastic gradient descent. As for SVM, [SVMLight](http://svmlight.joachims.org/)'s learning module(svm_learn) is used.

##Data Format:

The training and test file share the same format. Each line represents a document. The first integer in the line means whether the document is relevant to a query. Next follows the query id and the values for 44 features. 

For example, the line

0 qid:11 1:31.77 2:32.44

It means this document is irrelevant to query 11. The first feature value is 31.77. The second feature value is 32.44.

##Output
The program reads a test set file and produce a raking file in the following format:
ranking score for document1
ranking score for document2
....

