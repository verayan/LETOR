import math
import argparse
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default="data.txt", help="the data file")
args =  parser.parse_args()

def normalize(row):
    s = math.sqrt(reduce(lambda x,y: x + y*y,row))
    return map(lambda x : x/s, row)

# generate feature pairs for training set
def transform_train_data(file,outfile):
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
    with open(outfile,'w') as f:
        for row in feature_set:
            for irrelevant in feature_set[row][0]:
                for relevant in feature_set[row][1]:
                    row_data =  map(lambda x,y:x-y,relevant,irrelevant)
                    row_data = ["%d:%f" %(1 + x[0],x[1]) for x in enumerate(row_data)]
                    line = ' '.join(['+1'] + row_data)
                    f.write("%s\n" % line)


def transform_test_data(infile,outfile):
    features = []
    labels = []
    with open(infile,"r")as f:
        rows = f.readlines()
    for row in rows:
        values = row.split(" ")
        label = str(int(values[0]))
        labels.append(label)
        feature_list = []
        for value in values[2:-3]:
            feature_list.append(float(value.split(":")[1]))
        features.append(normalize(feature_list))
    feature_mat = np.array(features)
    with open(outfile,"w") as f:
        for i in xrange(len(labels)):
            line = labels[i]
            for j in xrange(feature_mat.shape[1]):
                line += " "+str(j+1) +":" +str(feature_mat[i,j])
            line += "\n"
            f.write(line)

def main():
    config = args.data
    with open(config, "r") as f:
        lines = f.readlines()
    train_file = lines[0].split("=")[1].strip("\n")
    test_file = lines[1].split("=")[1].strip("\n")
    c = float(lines[2].split("=")[1])
    new_train_file = "newtrain.txt"
    transform_train_data(train_file,new_train_file)
    new_test_file= "output.txt"
    transform_test_data(test_file,new_test_file)
    model_file = "model.txt"
    os.system("./implementation/svm_learn -c %f -b 0 %s %s" % (c,new_train_file,model_file))
    os.system("./implementation/svm_classify %s %s /dev/stdout"%(new_test_file,model_file))

if __name__ == "__main__":
    main()