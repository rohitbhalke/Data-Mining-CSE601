import numpy as np
import sys
import itertools


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


def getInitialFrequentAttributeSet(data, support, K, support_Map):
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
            support_Map[key] = map[key]
    return attribute_set, support_Map


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

def getFrequentItemSets(frequent_itemsets, support, K, originalDataSet, support_Map):
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
                    support_Map[','.join(combination)] = support_count
        return next_frequnet_itemSets, support_Map
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
                        support_Map[','.join(combination)] = support_count
        return next_frequnet_itemSets, support_Map



def generateItemSets(dataList, support, originalDataSet):
    #originalDataSet = dataList
    most_frequent_itemsets = {}
    support_Map = {}      # key value -> attbset -> frequency
    K = 1
    frequent_itemsets, support_Map = getInitialFrequentAttributeSet(dataList, support, K, support_Map)
    frequency_previous_frequent_itemsets = len(frequent_itemsets)
    most_frequent_itemsets[K] =  frequent_itemsets

    while frequency_previous_frequent_itemsets > 0 :
        K = K + 1
        #print(K)
        frequent_itemsets,support_Map = getFrequentItemSets(frequent_itemsets, support, K, originalDataSet, support_Map)
        frequency_previous_frequent_itemsets = len(frequent_itemsets)
        if frequency_previous_frequent_itemsets > 0 :
            most_frequent_itemsets[K] = frequent_itemsets

    return most_frequent_itemsets, support_Map


def getAllLengthSubsets(items):
    return_subsets = []
    subsets = []
    list_Items = list(items.split(","))
    for i in range(1, len(list_Items)):
        result = list(itertools.combinations(list_Items, i))
        for tuple in result:
            temp = []
            for attb in tuple:
                temp.append(attb)
            #temp = np.array(temp)
            subsets.append(temp)
        #print(result)

    # for s in subsets:
    #     print(s)
    #subsets = np.array(subsets)
    #subsets = np.unique(subsets, axis=0)
    return subsets



# generate rules
def generateRules(most_frequent_itemsets, support_Map, confidence):

    rules = []
    for key in most_frequent_itemsets:
        if key > 1:
            itemsets = most_frequent_itemsets[key]

            for item in itemsets:
                item_List = item.split(",")
                subsets = getAllLengthSubsets(item)

                for candidate in subsets:
                    candidate_Str = ','.join(candidate)
                    if support_Map[candidate_Str]/support_Map[item] >= confidence:
                        head = candidate
                        body = list(set(item_List) - set(candidate))
                        rule = []
                        rule.append(head)
                        rule.append(body)
                        # rules would be list of 2 lists,
                        # 1. Head
                        # 2. Body
                        rules.append(rule)
    filtered = []
    generated = {}
    for item in rules:
        first = ",".join(item[0])
        second =  ",".join(item[1])
        output = first + "->"+second
        print(output)


        if not output in generated:
            print(output)
            filtered.append(item)
            generated[output] = 1

    # print(len(filtered))
    # print(filtered)
    return filtered


def main(filePath, supportPercentage):
    dataset = readFile(filePath)
    support = (supportPercentage*len(dataset))/100
    mergedDataSetInSingleList =  np.concatenate(dataset, axis=0)       # Merge array of array in single array
    most_frequent_itemsets, support_Map = generateItemSets(mergedDataSetInSingleList, support, dataset)
    for key in most_frequent_itemsets:
        print(key,len(most_frequent_itemsets[key]),most_frequent_itemsets[key])

    rules = generateRules(most_frequent_itemsets, support_Map, 70/100)
    print("rules for ", supportPercentage)
    print("length of rules ", len(rules))

    print(len(rules))


filePath = "associationruletestdata.txt"

for supportPercentage in [50]:
    print("Most_Frequent_Attribute_Sets for Support : ", supportPercentage)
    main(filePath, supportPercentage)


