# Learning to rank

This project is about training ranking models so that top ranked documents relevant to queries can be returned.

The data set used is TREC dataset in LETOR from MSA. There are 44 features associated with each document including BM25, tf-idf and pagerank. More detailed info about the dataset can be found in Readme_TREC_dataset.pdf

To train pair-wise LETOR model, the classification model(logistic regression and SVM) are used to find the best parameter values. A linear combination of features weighted by these parameter values can be used to give scores for webpages based on their relevance.

Data Format:
The training and test file share the same format.Each line represents a document. The first integer in the line means whether the document is relevant to a query.Next follows the query id and the values for 44 features. 
For example, the line
0 qid:11 1:31.77 2:32.44
It means this document is irrelevant to query 11. It has 31.77 on first feature , 32.44 on second feature..
