import numpy as np
import re
import pandas as pd
import random
from itertools import accumulate
import math

tennis_ATTR_LENGTH = [3, 3, 2, 2, 2]

class GA(object):
    """
    genetic algorithm, support basic operations: select, mutate, crossover 
    select strategy: fitness-proportional, tournament, rank
    convert hypothese to readable rules
    """
    
    def __init__(self, p, numOfRulesPerHypo, numofattr, numofoutput, r, m, numOfChangeBitsPerRule, fit_threshold, maxGeneration, strategy, dataType):
        self.population = p
        self.numOfRulesPerHypo = numOfRulesPerHypo
        self.numOfAttr = numofattr
        self.numOfOutput = numofoutput
        self.replacementRate = r
        self.mutationRate = m
        self.numOfChangeBitsPerRule = numOfChangeBitsPerRule
        self.fit_threshold = fit_threshold
        self.maxGeneration = maxGeneration
        self.strategy = strategy
        self.dataType = dataType
        self.crossoverPop = int(r * p / 2)
        self.selectionPop = p - self.crossoverPop * 2 
        self.mutationPop = int(m*p)

# generate random hypos with population p
        initPopulation = []
        for i in range (p):
            # binary format input of one rule
            chrom1 = np.random.randint(2, size=self.numOfAttr)
            # binary format output of one rule
            chrom2 = np.zeros(self.numOfOutput).astype(int)
            chrom2[random.randint(0, self.numOfOutput - 1)] = 1
            chrom1 = np.append(chrom1, chrom2)
            for i in range(self.numOfRulesPerHypo-1):
                chrom3 = np.random.randint(2, size=self.numOfAttr)
                chrom4 = np.zeros(self.numOfOutput).astype(int)
                chrom4[random.randint(0, self.numOfOutput - 1)] = 1
                chrom3 = np.append(chrom3, chrom4)
                chrom1 = np.append(chrom1, chrom3)
            initPopulation.append(chrom1)

        initPopArray = np.array(initPopulation)
        self.initPopulation = initPopArray
# check if that hypo is an invalid hypo
    def valid(self, hypo):
        if self.dataType == 'tennis':
            sumOftennisAttr = sum(tennis_ATTR_LENGTH)
            numOfRule = int(len(hypo)/sumOftennisAttr)
            for h in range(numOfRule):
                rule = hypo[h*sum(tennis_ATTR_LENGTH):(h+1)*sum(tennis_ATTR_LENGTH)]
                sum1 = 0
                sum3 = 0
                for i in range (len(tennis_ATTR_LENGTH)-1):
                    for j in range (sum3, sum3+tennis_ATTR_LENGTH[i], 1):
                        sum1 = sum1 + rule[j]
                    if sum1 == 0:
                        # invalid when none of attribute is used
                        return False
                    sum1 = 0
                    sum3 = sum3 + tennis_ATTR_LENGTH[i]
                # if sum of output bits is not one then false
                sum2 = 0
                for n in range (tennis_ATTR_LENGTH[-1]):
                    sum2 = sum2 + rule[len(rule) - n -1]
                if sum2 != 1:
                    return False
            return True
        elif self.dataType == 'iris':
            lenOfOneRule = self.numOfAttr + self.numOfOutput
            numOfRule = self.numOfRulesPerHypo
            # if the output is not 100, 010 or 001 then false
            for h in range(numOfRule):
                sum1 = 0
                for n in range (self.numOfOutput):
                    sum1 = sum1 + hypo[h*lenOfOneRule-n-1]
                if sum1 != 1:
                    return False
            return True

    def hasConverged(self, correct, generationCount):
        if max(correct) < self.fitThreshold:
            return True
        if self.generationCount == self.maxGenerations:
            return True
  
    def onePointCrossover(self, hypo1, hypo2):
        crossoverIndex = randint(0, len(hypo1))
##        print('CrossoverIndex: ' + str(CrossoverIndex))
        selfLeftHalf = hypo1[:CrossoverIndex]
        selfRightHalf = hypo1[CrossoverIndex:]
##        print('I1: ' + SelfLeftHalf + '   ' + SelfRightHalf)
        otherLeftHalf = hypo2[:CrossoverIndex]
        otherRightHalf = hypo2[CrossoverIndex:]
##        print('I2: ' + OtherLeftHalf + '   ' + OtherRightHalf)
        newHypo1 = np.concatenate((selfLeftHalf, otherRightHalf), axis = 0) 
        newHypo2 = np.concatenate((otherLeftHalf, selfRightHalf), axis = 0)
        return newHypo1, nwHypo2
    
    
    def twoPointCrossover(self):
        pass

    def applyCrossover(self, hypo1, hypo2, NumberOfCrossoverPoints=1):
        ## Applies Crossover to self and to OtherInvidual, editing their Genotype
        if NumberOfCrossoverPoints == 1:
            self.onePointCrossover(hypo1, hypo2)
        else:
            self.twoPointCrossover()
  
    def shouldApplyCrossover(self):
        RandomFraction = uniform(0, 1)
        if self.replacementRate >= RandomFraction:
            return True
        else: return False

    def shouldApplyMutation(self):
        RandomFraction = uniform(0, 1)
        if self.mutationRate >= RandomFraction:
            return True
 
    def binaryToDecimal(self, binaryArray):
        binaryArray = binaryArray.tolist()
        binaryArray = list(map(str, binaryArray))
        pointer = 3
        binaryArray.insert(pointer, '.')
        s = ''.join(binaryArray)
        return self.parse_bin(s)

    def parse_bin(self, s):
        t = s.split('.')
        # print('parse bin')
        # print (s)
        return int(t[0], 2) + int(t[1], 2) / 2. ** len(t[1])

    def binaryToDecimalRule(self, hypo): 
        lenOfOneRule = self.numOfAttr + self.numOfOutput
        numOfRule = self.numOfRulesPerHypo
        rulesInOneHypo = []
        for r1 in range (numOfRule):
            onerule = hypo[r1*lenOfOneRule:(r1+1)*lenOfOneRule]
            rule = []
            for attrIndex in range (8):
                rule.append(self.binaryToDecimal(onerule[attrIndex*int(self.numOfAttr/8):(attrIndex+1)*int(self.numOfAttr/8)]))
            rule.append(onerule[-3])
            rule.append(onerule[-2])
            rule.append(onerule[-1])
            rulesInOneHypo.append(rule)
        return rulesInOneHypo

    def calFitness(self, hypo, x):
        if self.dataType == 'tennis':
            sumOftennisAttr = sum(tennis_ATTR_LENGTH)
            numOfRule = int(len(hypo) / sumOftennisAttr)
            fitness = 0
            if not self.valid(hypo):
                return 0, hypo
            for h in range(numOfRule):
                rule = hypo[h * sumOftennisAttr:(h + 1) * sumOftennisAttr]
                numOfMatch = 0
                for i in range(len(x)-tennis_ATTR_LENGTH[-1]):
                    if x[i] == 1 and rule[i] == 1:
                        numOfMatch = numOfMatch+1
                # check predict output
                if numOfMatch == len(tennis_ATTR_LENGTH)-1:
                    for outputIndex in range (tennis_ATTR_LENGTH[-1]):
                        if rule[len(rule)-outputIndex-1] != x[len(rule)-outputIndex-1]:
                            return fitness, hypo
                    fitness = 1
                else:
                    continue
            return fitness, hypo
        elif self.dataType == 'iris':
            rulesInOneHypo = self.binaryToDecimalRule(hypo)
            fitness = 0
            negativeFitness = 0
            for i in range (len(rulesInOneHypo)):
                rule = rulesInOneHypo[i]
                numOfMatch = 0
                for j in range (4):
                    if float(x[j]) > rule[j*2] and float(x[j]) < rule[j*2+1]:
                        numOfMatch = numOfMatch + 1
                    elif float(x[j]) < rule[j*2] and float(x[j]) > rule[j*2+1]:
                        numOfMatch = numOfMatch + 1
                if numOfMatch == 4:
                    if float(x[-3]) == rule[-3] and float(x[-2]) == rule[-2] and float(x[-1]) == rule[-1]:
                        fitness = 1
                        return fitness, rulesInOneHypo
            return fitness, rulesInOneHypo

# calculate the fitness for all hypos on all instancse
    def allFitness(self, hypos, X):
        correctes = []
        for hypo in hypos:
            correct = 0
            for x in X:
                oneFitness, rules = self.calFitness(hypo, x)
                correct = correct + oneFitness
            correctes.append(correct)
        # correct rate equal to correct predictions over number of instances
        correctes2 = [cor/X.shape[0] for cor in correctes]
        # GABIL fitness equation
        fitnesses = [i**2 for i in correctes2]
        return fitnesses, correctes2

# accumulate the probability and pick one hypo
    def weighted_random_choice(self, fitnesses):
        max = sum(fitnesses)
        if sum(fitnesses) == 0:
            wheel = list([1/len(fitnesses) for i in fitnesses])
            accumulate_wheel = list(accumulate(wheel))
        else:
            wheel = list([i / max for i in fitnesses])
            accumulate_wheel = list(accumulate(wheel))
        pick = random.uniform(0, sum(wheel))
        for index in range (len(wheel)):
            if pick < accumulate_wheel[index]:
                return accumulate_wheel[index], index

# select num high probabilty hypos
    def fitnessPropotionSelection(self, hypos, fitnesses, num):
        sortedHypos = []
        hypos2 = hypos
        fites2 = np.array(fitnesses)
        while num != len(sortedHypos):
            whe, index = self.weighted_random_choice(fites2)
            sortedHypos.append(hypos2[index])
        return np.array(sortedHypos)

# select num high probabilty hypos using tournament selection method
    def tournamentSelection(self, hypos, fitnesses, num):
        sortedHypos = []
        hypos2 = hypos
        fites2 = np.array(fitnesses)
        selectedNum = 0
        canSelectNum = num
        ranNum = []
        while num-2 != selectedNum:
            if canSelectNum == 1:
                break
            ranNum = random.sample(range(0, canSelectNum),2)
            if fites2[ranNum[0]] >= fites2[ranNum[1]]:
                swapHypo = hypos2[len(hypos2) - selectedNum-1]
                hypos2[len(hypos2) - selectedNum-1] = hypos[ranNum[0]]
                hypos[ranNum[0]] = swapHypo
            else:
                swapHypo = hypos2[len(hypos2) - selectedNum-1]
                hypos2[len(hypos2) - selectedNum-1] = hypos[ranNum[1]]
                hypos[ranNum[1]] = swapHypo
            selectedNum = selectedNum + 1
            canSelectNum = canSelectNum - 1
        hypos2 = hypos2.tolist()
        hypos3 = hypos2[::-1]
        hypos3 = np.array(hypos3)
        return hypos3

# accumulate the probability generated by rank and pick one hypo
    def rank_random_choice(self, fitnesses):
        rank_index_fites2 = ss.rankdata(fitnesses)
        summary = sum(rank_index_fites2)
        rank_prob = [i/summary for i in rank_index_fites2]
        accumulate_wheel = list(accumulate(rank_prob))
        pick = random.uniform(0, sum(rank_prob))
        for index in range (len(accumulate_wheel)):
            if pick < accumulate_wheel[index]:
                return accumulate_wheel[index], index

# select num high probabilty hypos using rank selection
    def rankSelection(self, hypos, fitnesses, num):
        sortedHypos = []
        hypos2 = hypos
        fites2 = np.array(fitnesses)
        while num != len(sortedHypos):
            whe, index = self.rank_random_choice(fites2)
            sortedHypos.append(hypos2[index])
            # hypos2 = np.delete(hypos2, index, axis=0)
            # fites2 = np.delete(fites2, index, axis=0)
        # for hys in hypos2:
            # sortedHypos.append(hys)
        return np.array(sortedHypos)

# select high fitness hypos by different strategies
    def selectStrategy(self, XY):
        if self.strategy == 'fitness-proportional':
            fit, correct = self.allFitness(self.initPopulation, XY)
            return (self.fitnessPropotionSelection(self.initPopulation, fit, self.population))
            print('')
        elif self.strategy == 'tournament':
            fit, correct = self.allFitness(self.initPopulation, XY)
            return (self.tournamentSelection(self.initPopulation, fit, self.population))
            print('')
        elif self.strategy == 'rank':
            fit, correct = self.allFitness(self.initPopulation, XY)
            return (self.rankSelection(self.initPopulation, fit, self.population))
            print('')

    def shouldApplyCrossover(self):
        RandomFraction = uniform(0, 1)
        if self.replacementRate >= RandomFraction:
            return True
        else: return False

    def shouldApplyMutation(self):
        RandomFraction = uniform(0, 1)
        if self.mutationRate >= RandomFraction:
            return True

# single point crossover at middle position
    def crossoverBase(self, hypo1, hypo2):
        length = len(hypo1)
        cross_over_point = int(length/2)
        # cross_over_point = np.random.randint(length, size=1)[0]
        newhypo1 = np.concatenate((hypo1[0:cross_over_point], hypo2[cross_over_point:]), axis=0)
        newhypo2 = np.concatenate((hypo2[0:cross_over_point], hypo1[cross_over_point:]), axis=0)
        return newhypo1, newhypo2

# mutate a multi-rules hypo without changing output bits
    def mutationHypo(self, hypo):
        changeIndex = np.random.randint(self.numOfAttr, size=(self.numOfRulesPerHypo, self.numOfChangeBitsPerRule))
        lengthOfRule = self.numOfAttr + self.numOfOutput
        for i in range (self.numOfRulesPerHypo):
            # rule = hypo[i*lengthOfRule:(i+1)*lengthOfRule]
            for index in range (changeIndex.shape[1]):
                if hypo[i*lengthOfRule + changeIndex[i][index]] == 0:
                    hypo[i*lengthOfRule + changeIndex[i][index]] = 1
                elif hypo[i*lengthOfRule + changeIndex[i][index]] == 1:
                    hypo[i*lengthOfRule + changeIndex[i][index]] = 0
        return hypo

    def mutationBase(self, hypo):
        outputLength = self.numOfRulesPerHypo*self.numOfOutput
        randomNumber = random.randint(0, len(hypo)-1-outputLength)
        poi = int (randomNumber/self.numOfAttr)
        randomPoint = randomNumber + poi*self.numOfOutput
        if hypo[randomPoint] == 0:
            hypo[randomPoint] = 1
        elif hypo[randomPoint] == 1:
            hypo[randomPoint] = 0
        return hypo

# covert one hypo to human readable rules
    def printRules(self, bestHypo):
        if self.dataType == 'tennis':
            sumOftennisAttr = sum(tennis_ATTR_LENGTH)
            numOfRule = self.numOfRulesPerHypo
            tennis_attr = pd.read_csv("data/tennis/tennis-attr.txt", sep=' ', header=None)
            for h in range(numOfRule):
                rule = bestHypo[h * sumOftennisAttr:(h + 1) * sumOftennisAttr]
                ruleString = []
                rulePointer = 0
                firstTime1 = 0
                for num in range (len(tennis_ATTR_LENGTH)-1):
                    if firstTime1 != 1:
                        firstTime1 = 1
                    else:
                        ruleString.append('  ^  ')
                    ruleString.append(tennis_attr.iloc[num,0])
                    ruleString.append('=')
                    firstTime2 = 0
                    for index in range (1,tennis_ATTR_LENGTH[num]+1,1):
                        if rule[rulePointer] == 1:
                            if firstTime2 != 1:
                                firstTime2 = 1
                            else:
                                ruleString.append('|')
                            ruleString.append(tennis_attr.iloc[num,index])
                        rulePointer = rulePointer + 1
                outputString = []
                outputString.append(tennis_attr.iloc[-1,0])
                outputString.append('=')
                for outputIndex in range (tennis_ATTR_LENGTH[-1]):
                    if rule[rulePointer] == 1:
                        outputString.append(tennis_attr.iloc[-1,outputIndex+1])
                    rulePointer = rulePointer + 1  
                s = ''.join(ruleString) 
                t = ''.join(outputString)
                print ( s + " -> " + t)
        
        elif self.dataType == 'iris':
            rules = self.binaryToDecimalRule(bestHypo)
            attrString = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width']
            outputAttr = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
            for i in range (len(rules)):
                ruleString = []
                rule = rules[i]
                for j in range (4):
                    if rule[2*j] < rule[2*j+1]:
                        ruleString.append(str(rule[2*j]))
                        ruleString.append('<')
                        ruleString.append(attrString[j])
                        ruleString.append('<')
                        ruleString.append(str(rule[2*j+1]))
                        if j != 3:
                            ruleString.append('  and  ')
                    elif rule[2*j] > rule[2*j+1]:
                        ruleString.append(str(rule[2*j+1]))
                        ruleString.append('<')
                        ruleString.append(attrString[j])
                        ruleString.append('<')
                        ruleString.append(str(rule[2*j]))
                        if j != 3:
                            ruleString.append('  and  ')
                outputString = []
                for outputIndex in range (3):
                    if rule[-1-outputIndex] == 1:
                        outputString.append(outputAttr[outputIndex])
                s = ''.join(ruleString) 
                t = ''.join(outputString)
                print (s + "=>" + t)
 

    def fit(self, XY, outputGen=False):
        if self.dataType == 'tennis':
            fit, correct = self.allFitness(self.initPopulation, XY)
            correct_list = []
            hypo_list = []
            generationNum = 1
            while max(correct) < self.fit_threshold:
                if generationNum == self.maxGeneration:
                    break
                # selection
                selectedHypos1 = self.selectStrategy(XY)
                p_springoff = []
                p_springoff = selectedHypos1[0:self.selectionPop,]
                p_springoff = p_springoff.tolist()
                # cross over
                selectedHypos2 = self.selectStrategy(XY)
                for i in range (self.crossoverPop):
                    a, b = self.crossoverBase(selectedHypos2[2*i], selectedHypos2[2*i+1])
                    p_springoff.append(a)
                    p_springoff.append(b)
                psArray = np.array(p_springoff)
                # mutation
                mutationList = random.sample(range(1, self.population), self.mutationPop)
                for mut in mutationList:
                    mutatedHypo = self.mutationBase(psArray[mut])
                    # generate a valid hypo
                    while self.valid(mutatedHypo) == False:
                        mutatedHypo = self.mutationHypo(psArray[mut])
                    psArray[mut] = mutatedHypo
                self.initPopulation = psArray
                fit, correct = self.allFitness(self.initPopulation, XY)
                # print('correct rate for each generation')
                # print(correct)
                # record hypo and correct rate if needed
                if outputGen and (generationNum%10==0):
                    correct_list.append(max(correct))
                    bestHypoIndex = correct.index(max(correct))
                    hypo_list.append(self.initPopulation[bestHypoIndex]) 
                    print ("Generation: " + str(generationNum))
                    print ("Accuracy: " + str(max(correct))) 
                generationNum = generationNum + 1
            if outputGen:
                return correct_list, hypo_list
            else:
                bestHypoIndex = correct.index(max(correct))
                fit, correct = self.allFitness(np.expand_dims(self.initPopulation[bestHypoIndex], axis=0), XY)
                return correct, self.initPopulation[bestHypoIndex]
        elif self.dataType == 'iris':
            fit, correct = self.allFitness(self.initPopulation, XY)
            correct_list = []
            hypo_list = []
            generationNum = 1
            while max(correct) < self.fit_threshold:
                # break if 
                if generationNum == self.maxGeneration:
                    break
                # selection
                selectedHypos1 = self.selectStrategy(XY)
                p_springoff = []
                p_springoff = selectedHypos1[0:self.selectionPop,]
                p_springoff = p_springoff.tolist()
                # cross over
                selectedHypos2 = self.selectStrategy(XY)
                for i in range (self.crossoverPop):
                    a, b = self.crossoverBase(selectedHypos2[2*i], selectedHypos2[2*i+1])
                    p_springoff.append(a)
                    p_springoff.append(b)
                psArray = np.array(p_springoff)
                # mutation
                mutationList = random.sample(range(1, self.population), self.mutationPop)
                for mut in mutationList:
                    mutatedHypo = self.mutationBase(psArray[mut])
                    # generate a valid hypo
                    while self.valid(mutatedHypo) == False:
                        mutatedHypo = self.mutationHypo(psArray[mut])
                    psArray[mut] = mutatedHypo
                self.initPopulation = psArray
                fit, correct = self.allFitness(self.initPopulation, XY)
                # print('correct rate for each generation')
                # print(correct)
                if outputGen and (generationNum%5==0):
                    correct_list.append(max(correct))
                    bestHypoIndex = correct.index(max(correct))
                    hypo_list.append(self.initPopulation[bestHypoIndex]) 
                    print ("Generation: " + str(generationNum))
                    print ("Accuracy: " + str(max(correct))) 
                generationNum = generationNum + 1
            # return best correct rate and best hypo
            if outputGen:
                return correct_list, hypo_list
            else:
                bestHypoIndex = correct.index(max(correct))
                fit, correct = self.allFitness(np.expand_dims(self.initPopulation[bestHypoIndex], axis=0), XY)
                return max(correct), self.initPopulation[bestHypoIndex]

    def predict(self, bestHypo, XY):
        fit, correct = self.allFitness(np.expand_dims(bestHypo, axis=0), XY)
        return correct
