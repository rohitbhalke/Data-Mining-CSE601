import numpy as np
import sys

def formatData(list):
    # avoid last array entry as it is diseases name
    temp = []
    for i in range(0, len(list)-1):
        value = list[i]
        temp.append("G"+str(i+1)+"_"+value)
    return temp

def readFile(path):
    data = []
    with open(path) as fp:
        line = fp.readline()
        while line:
            #print(line)
            list = line.replace("\n", "").split("\t")
            formattedData = formatData(list)
            data.append(formattedData)
            line = fp.readline()
    #print(data)
    return data


def getInitialFrequentAttributeSet(data, support, K):
    map = {}
    attribute_set = []
    for attribute in data:
        if attribute in map:
            map[attribute] = map[attribute] + 1
        else:
            map[attribute] = 1

    # iterate over map and delete those attributes which are less than support
    pruned_map = {}
    for key in map:
        if map[key] >= support:
            pruned_map[key] = map[key]
            attribute_set.append(key)
    return attribute_set


def getSupportCount(attributes, originalDataSet):
    count = 0
    for line in originalDataSet:
        flag = False
        for attb in attributes:
            if attb not in line:
                flag = True
                break
        if flag == False:
            count += 1
    return count

def getFrequentItemSets(frequent_itemsets, support, K, originalDataSet):
    next_frequnet_itemSets = []
    if K-2 <= 0:
        for i in range(0, len(frequent_itemsets)):
            for j in range(i+1, len(frequent_itemsets)):
                combination = []
                combination.append(frequent_itemsets[i])
                combination.append(frequent_itemsets[j])
                support_count = getSupportCount(combination, originalDataSet)
                if support_count >= support:
                    combination = np.sort(combination)
                    next_frequnet_itemSets.append(','.join(combination))
        return next_frequnet_itemSets
    else:
        for i in range(0, len(frequent_itemsets)):
            for j in range(i+1, len(frequent_itemsets)):
                combination = []
                first_itemsets = frequent_itemsets[i]
                second_itemsets = frequent_itemsets[j]
                #print(first_itemsets, second_itemsets)
                # check whether the K-2 elements are same or not, if yes then only create a new frequent itemset
                for itr in range (0, K-2):
                    considerItemSet = True
                    first_itemsets_List = first_itemsets.split(",")
                    second_itemsets_List = second_itemsets.split(",")
                    if(first_itemsets_List[itr]!=second_itemsets_List[itr]):
                        considerItemSet = False
                        break

                if considerItemSet == True:
                    combination = frequent_itemsets[i].split(",") + frequent_itemsets[j].split(",")
                    combination = np.unique(combination)
                    support_count = getSupportCount(combination, originalDataSet)
                    if support_count >= support:
                        combination = np.sort(combination)
                        next_frequnet_itemSets.append(','.join(combination))

        return next_frequnet_itemSets



def generateItemSets(dataList, support, originalDataSet):
    #originalDataSet = dataList
    most_frequent_itemsets = {}
    K = 1
    frequent_itemsets = getInitialFrequentAttributeSet(dataList, support, K)
    frequency_previous_frequent_itemsets = len(frequent_itemsets)
    most_frequent_itemsets[K] =  frequent_itemsets

    while frequency_previous_frequent_itemsets > 0 :
        K = K + 1
        #print(K)
        frequent_itemsets = getFrequentItemSets(frequent_itemsets, support, K, originalDataSet)
        frequency_previous_frequent_itemsets = len(frequent_itemsets)
        if frequency_previous_frequent_itemsets > 0 :
            most_frequent_itemsets[K] = frequent_itemsets

    return most_frequent_itemsets



def main(filePath, supportPercentage):

    dataset = readFile(filePath)
    support = (supportPercentage*len(dataset))/100
    mergedDataSetInSingleList =  np.concatenate(dataset, axis=0)       # Merge array of array in single array
    most_frequent_itemsets = generateItemSets(mergedDataSetInSingleList, support, dataset)
    for key in most_frequent_itemsets:
        print(key,len(most_frequent_itemsets[key]),most_frequent_itemsets[key])


filePath = "associationruletestdata.txt"

for supportPercentage in [60,70]:
    print("Most_Frequent_Attribute_Sets for Support : ", supportPercentage)
    main(filePath, supportPercentage)