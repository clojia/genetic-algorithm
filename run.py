import os
import argparse
import numpy as np
import pandas as pd

from ga import GA

def digitalize(train, test, attributes, dataType):
    if dataType == "tennis":
        merge = pd.concat([train, test])
        attributes_tr = attributes.transpose()
        numOfAttr = attributes_tr.shape[1] - 1
        attr_dic_list = []
        for i in range(numOfAttr + 1):
            attr_playT = attributes_tr[i].tolist()
            attr_playT = list(filter(None, attr_playT))
            if attr_playT == []:
                continue
            dic = dict((a, index - 1) for index, a in enumerate(attr_playT))
            attr_dic_list.append(dic)
        merge_dummies = pd.concat([pd.get_dummies(merge[col]) for col in train], axis=1, keys=train.columns)
        # print(merge_dummies)
        input_dummies = merge_dummies[merge_dummies.columns[:-2]].values
        output_dummies = merge_dummies[merge_dummies.columns[-2:]].values
        input_train = input_dummies[:10,:]
        input_test = input_dummies[10:,:]
        output_train = output_dummies[:10,:]
        output_test = output_dummies[10:,:] 
        return input_train, output_train, input_test, output_test
    
    elif dataType == "iris":
        attributes_tr = attributes.transpose()
        numOfAttr = attributes_tr.shape[1] - 1
        attr_dic_list = []
        for i in range(numOfAttr + 1):
            attr_playT = attributes_tr[i].tolist()
            attr_playT = list(filter(None, attr_playT))
            if attr_playT == []:
                continue
            dic = dict((a, index - 1) for index, a in enumerate(attr_playT))
            attr_dic_list.append(dic)
        pref = list(attr_dic_list[-1].keys())[1:]
        input_dummies = train[train.columns[:-1]].values
        output_dummies = pd.get_dummies(train[train.columns[-1]], prefix_sep = pref).values
        input_test = test[test.columns[:-1]].values
        output_test = pd.get_dummies(test[test.columns[-1]], prefix_sep = pref).values
        return input_dummies, output_dummies, input_test, output_test


'''
output the learned rules (in human-readable form similar to HW2), and
accuracy on training and test sets.
'''
def testTennis(trainFile, testFile, attrFile):
    tennis_train = pd.read_csv(trainFile, sep=" ", header=None)
    tennis_test = pd.read_csv(testFile, sep=" ", header=None)
    tennis_attr = pd.read_csv(attrFile, sep=" ", header=None)
    
    input, output, input_test, output_test = digitalize(tennis_train, tennis_test, tennis_attr,"tennis")
    input = np.concatenate((input, output), axis=1) 
    test_input = np.concatenate((input_test, output_test), axis=1)
    # p, numOfRulesPerHypo, numofattr, numofoutput, r, m, fit_threshold, stopGeneration, strategy, dataType):
    ga = GA(100, 4, 10, 2, 0.3, 0.1, 1, 1, 30, 'fitness-proportional', 'tennis')
    trainAcc, bestHypo = ga.fit(input)
    print ('Training data accuracy is: ' + str(trainAcc[0])) 
    testAcc = ga.predict(bestHypo, test_input)
    print ('Test data accuracy is: ' + str(testAcc[0]))
    ga.printRules(bestHypo)
  
'''
output the learned rules (in human-readable form similar to HW2), and accuracy
on training and test sets.
'''
def testIris(trainfile, testfile, attrfile):
    iris_train = pd.read_csv(trainfile, header=None)
    iris_test = pd.read_csv(testfile, header=None)
    iris_attr = pd.read_csv(attrfile, header=None)
    input, output, input_test, output_test = digitalize(iris_train, iris_test, iris_attr,"iris")
    input = input.astype(np.float)
    input = np.concatenate((input, output), axis=1)
    test_input = np.concatenate((input_test, output_test), axis=1)
    # p, numOfRulesPerHypo, numofattr, numofoutput, r, m, numOfChangeBitsPerRule, fit_threshold, stopGeneration, strategy, dataType): tournament fitness-proportional rank
    # ga = GA(100, 6, 28, 3, 0.4, 0.2, 1, 0.8, 100, 'rank', 'iris')
    # (6*2*4+3)*4  *4
    ga = GA(400, 6, 48, 3, 0.3, 0.1, 1, 1, 50, 'tournament', 'iris')
    correct, bestHypo = ga.fit(input)
    print ('correct rate on traning data' + str(correct))
    ga.printRules(bestHypo)
    cor = ga.predict(bestHypo, test_input)
 
def testIrisSelection(trainfile, testfile, attrfile):
    iris_train = pd.read_csv(trainfile, header=None)
    iris_test = pd.read_csv(testfile, header=None)
    iris_attr = pd.read_csv(attrfile, header=None)
    input, output, input_test, output_test = digitalize(iris_train, iris_test, iris_attr,"iris")
    input = input.astype(np.float)
    input = np.concatenate((input, output), axis=1)
    test_input = np.concatenate((input_test, output_test), axis=1)
    # stopGen = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    strategies = ['fitness-proportional', 'tournament', 'rank']
    cor_list = []
    strategy_list = []
    for j in range (len(strategies)):
        ga = GA(100, 6, 48, 3, 0.3, 0.1, 1, 1, 50, strategies[j], 'iris')
        print ('Strategy: ' + strategies[j]) 
        cor, bestHypo = ga.fit(input, outputGen=True)
        for i in range (len(cor)):
            cor2 = ga.predict(bestHypo[i], test_input)
            cor_list.append(cor2)
        cor_array = np.array(cor_list)
        strategy_list.append(cor_array)
        cor_list = []
    print ('correct rate for fitness-proportional, tournament, rank strategy at vary generations are')
    print (strategy_list)
   
def testIrisReplacement(trainfile, testfile, attrfile):
    iris_train = pd.read_csv(trainfile, header=None)
    iris_test = pd.read_csv(testfile, header=None)
    iris_attr = pd.read_csv(attrfile, header=None)
    input, output, input_test, output_test = digitalize(iris_train, iris_test, iris_attr,"iris")
    input = input.astype(np.float)
    input = np.concatenate((input, output), axis=1)
    test_input = np.concatenate((input_test, output_test), axis=1)
    replaceRate = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    strategies = ['fitness-proportional', 'tournament', 'rank']
    cor_list = []
    strategy_list = []
    for j in range (len(strategies)):
        print ('Strategy: ' + strategies[j])
        for i10 in range(1, 10, 1):
            i = i10 / 10
            print('Replacement rate: ' + str(i))
            ga = GA(100, 6, 48, 3, i, 0.1, 1, 1, 30, strategies[j], 'iris') 
            cor, bestHypo = ga.fit(input)
            testAcc = ga.predict(bestHypo, test_input)
            print("Accuracy: " + str(testAcc) +"\n" )
            cor_list.append(testAcc)
        cor_array = np.array(cor_list)
        strategy_list.append(cor_array)
        cor_list = []
    print ('correct rate for fitness-proportional, tournament, rank strategy at vary replacement rate are')
    print (strategy_list)
   

def main():
    parser = argparse.ArgumentParser(description="Genetic Algorithm")
    parser.add_argument("-e", "--experiment", required=True, dest="experiment", 
            choices=["testTennis", "testIris", "testIrisSelection", "testIrisReplacement"], help='experiment name.')
    args = parser.parse_args()
    print("Experiment: " + args.experiment)
    if args.experiment == "testTennis":
        testTennis("data/tennis/tennis-train.txt", "data/tennis/tennis-test.txt", "data/tennis/tennis-attr.txt")
    elif args.experiment == "testIris":
        testIris("data/iris/iris-train.txt", "data/iris/iris-test.txt", "data/iris/iris-attr.txt")
    elif args.experiment == "testIrisSelection":
        testIrisSelection("data/iris/iris-train.txt", "data/iris/iris-test.txt", "data/iris/iris-attr.txt") 
    elif args.experiment == "testIrisReplacement":
        testIrisReplacement("data/iris/iris-train.txt", "data/iris/iris-test.txt", "data/iris/iris-attr.txt")  

if __name__ == '__main__':
    main()
