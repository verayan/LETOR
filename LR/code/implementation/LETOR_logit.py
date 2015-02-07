import math
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default="data.txt", help="data file")
parser.add_argument("--rate", type=float, default=0.0001, help="learning rate")
parser.add_argument("--threshold", type = float,default = 4, help = "threshold")
args =  parser.parse_args()


# initialize the logistic regression model
# set parameter values to be all zero initially
def initialize_models(numFeatures):
    return np.zeros((1,numFeatures))

# training process
def gradient_descent(model, x, labels, threshold, rate, c):
    current_prob = None
    count = 0
    while True:
        estimated = ((x.dot(model.T)) * (-1))
        estimated_prob = 1 / (1 + np.exp(estimated))
        # add regularization
        gradient = x.T.dot(labels - estimated_prob) - c * model.T
        gradient[0] = gradient[0] + c * model.T[0]
        new_model = model + rate * gradient.T
        if current_prob != None and max_function(current_prob, model,
        estimated_prob, new_model, c, labels,threshold)  == True:
            break
        model = new_model
        current_prob = estimated_prob
        count = count + 1
    return model



# rule of stopping updating parameter values
def max_function(prob1, model1, prob2, model2, c, labels,threshold):
    max1 = np.sum(np.multiply(np.log(prob1), labels) +
           np.multiply((1 - labels), np.log(1 - prob1))) - 0.5 * c * (
           model1.dot(model1.T))[0, 0]
    max2 = np.sum(np.multiply(np.log(prob2), labels) +
           np.multiply((1 - labels), np.log(1 - prob2))) - 0.5 * c * (
           model2.dot(model2.T))[0, 0]
    return ((max2 - max1) < threshold or max2< max1)



# normalize each feature set
def normalize(row):
    s = math.sqrt(reduce(lambda x,y: x + y*y,row))
    return map(lambda x : x/s, row)


# read training data
# return data matrix and class label
# for each query
def read_train_data(file):
    feature_set = {}
    with open(file, "r") as f:
        rows = f.readlines()
    for row in rows:
        values = row.split(" ")
        feature = []
        for value in values[2:-3]:
            feature.append(float(value.split(":")[1]))
        feature = normalize(feature)
        sign = int(values[0])
        queryid = int(values[1].split(":")[1])
        if queryid not in feature_set:
            feature_set[queryid] = [[],[]]
        feature_set[queryid][sign].append(feature)
    total_labels = []
    pairwise_feature = []
    for row in feature_set:
        for irrelevant in feature_set[row][0]:
            for relevant in feature_set[row][1]:
                pairwise_feature.append([1] + map(lambda x,y:x-y,irrelevant,relevant))
                total_labels.append(0)
                pairwise_feature.append([1] + map(lambda x,y:y-x,irrelevant,relevant))
                total_labels.append(1)
    return np.array(pairwise_feature),np.array(total_labels)



def read_test_data(file):
    # construct a feature matrix without labels and the queryid
    features = []
    with open(file,"r") as f:
       rows = f.readlines()
    for row in rows:
        values = row.split(" ")
        feature_list = []
        for value in values[2:-3]:
            feature_list.append(float(value.split(":")[1]))
        features.append(normalize(feature_list))
    feature_mat = np.array(features)
    bias_terms = np.ones((feature_mat.shape[0],1))
    feature_mat = np.concatenate((bias_terms,feature_mat),axis=1)
    return feature_mat



def main():
    # read the configuration file
    config = args.data
    with open(config, "r") as f:
        lines = f.readlines()
    train_file = lines[0].split("=")[1].strip("\n")
    test_file = lines[1].split("=")[1].strip("\n")
    c = float(lines[2].split("=")[1])
    (total_feature_matrix,total_labels) = read_train_data(train_file)
    initial_model = initialize_models(total_feature_matrix.shape[1])
    total_labels = total_labels.reshape((total_labels.shape[0],1))
    finalmodel = gradient_descent(initial_model, total_feature_matrix,
                 total_labels, args.threshold, args.rate, c)
    test_features = read_test_data(test_file)
    scores = np.dot(test_features,finalmodel.T)
    # predict the relevance score on the test set
    for i in xrange(scores.shape[0]):
        print scores[i,0]



if __name__ == "__main__":
    main()
