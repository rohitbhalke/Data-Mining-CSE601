import pandas as pd
from itertools import combinations as c

data = pd.read_csv('associationruletestdata.txt', delimiter="\t", header=None)
n_columns = len(data.columns)


def process_data():
    i = 1
    for column in data:
        s = 'G' + str(i) + '_'
        if i < n_columns:
            data[column] = s + data[column].astype(str)
        i = i + 1


def is_item_set_frequent(item_set, support_threshold):

    cnt = 0
    for r in range((data.shape[0])):
        row_list = list(data.iloc[r, :])
        if set(item_set).issubset(set(row_list)):
            cnt = cnt + 1

    if cnt >= support_threshold:
        return True
    else:
        return False


def generate_candidate_items(frequent_item_list, minsup, total_freq_itemsets):

    k = 2
    candidate_items = list(c(frequent_item_list, k))

    f_list = []
    for item in candidate_items:
        list(item).sort()
        if is_item_set_frequent(item, minsup):
            f_list.append(item)
    f_list.sort()
    total_freq_itemsets += len(f_list)
    print("Number of length-" + str(k) + " frequent itemsets: " + str(len(f_list)))

    if len(f_list) != 0:
        frequent_item_list = f_list
        k += 1
        while True:
            f_list = []
            same_elems = k - 2
            for i in range(len(frequent_item_list) - 1):
                for j in range(i+1, len(frequent_item_list)):
                    include = True
                    for m in range(same_elems):
                        if frequent_item_list[i][m] != frequent_item_list[j][m]:
                            include = False
                            break
                    if include:
                        l1 = list(frequent_item_list[i])
                        l2 = list(frequent_item_list[j])
                        l1 = l1 + list(set(l2) - set(l1))
                        l1.sort()
                        if is_item_set_frequent(l1, minsup):
                            f_list.append(l1)
            f_list.sort()
            total_freq_itemsets += len(f_list)
            print("Number of length-" + str(k) + " frequent itemsets: " + str(len(f_list)))

            if len(f_list) == 0:
                break

            frequent_item_list = f_list
            k += 1

    print("Number of all lengths frequent itemsets: " + str(total_freq_itemsets))
    print()


def main():
    process_data()
    support_percent_list = [30, 40, 50, 60, 70]
    total_transactions = data.shape[0]

    for minsup in support_percent_list:
        total_freq_itemsets = 0
        support_threshold = (minsup / 100) * total_transactions
        frequent_item_list = []
        i = 1
        while i < n_columns:
            s1 = 'G' + str(i) + '_Up'
            s2 = 'G' + str(i) + '_Down'
            s1_list = [s1]
            s2_list = [s2]
            if is_item_set_frequent(s1_list, support_threshold):
                frequent_item_list.append(s1)
            if is_item_set_frequent(s2_list, support_threshold):
                frequent_item_list.append(s2)
            i += 1
        print("------------------------------------------")
        print("Support is set to be " + str(minsup) + "%")
        print("------------------------------------------")
        total_freq_itemsets = len(frequent_item_list)
        print("Number of length-1 frequent itemsets: " + str(len(frequent_item_list)))
        frequent_item_list.sort()
        generate_candidate_items(frequent_item_list, support_threshold, total_freq_itemsets)


if __name__ == '__main__':
    main()

