import numpy as np
import sys
import matplotlib.pyplot as plt
from Node import Node as Node
from PerformanceMetrics import MetricsCalculator
import json

class DecisionTree:

    def __init__(self):
        self.attribute_indexex = []
        self.COUNT = 10
        self.categorical_columns_index = []

    def read_file(self, path):
        data = []
        with open(path) as fp:
            line = fp.readline()
            while line:
                list = line.replace("\n", "").split("\t")
                data.append(list)
                line = fp.readline()
        data = np.array(data)
        return data

    def divide_data(self, data):
        train_data_percent = self.config['training_percentage']
        no_of_records = int(data.shape[0] * (train_data_percent/100))
        train_data = data[0:no_of_records]
        test_data = data[no_of_records:]
        return train_data, test_data

    # def divide_data(self,data):
    #     k = self.config['k_fold_validation']
    #     no_of_records = int(data.shape[0] / k)
    #     data_split = {}
    #     for i in range(0,k-1):
    #         start = i*no_of_records
    #         end = start + no_of_records
    #         data_split[i] = data[start:end]
    #     data_split[i+1] = data[end:]
    #     return data_split

    def get_gini_index(self, data):
        # find from data, how many are 1's and 0's,
        # last index is the prediction value
        total_records = data.shape[0]
        if total_records == 0 :
            return 0
        prediction_value_index = data.shape[1] - 1
        ones_count = 0
        zeros_count = 0
        for record in data:
            if int(record[prediction_value_index]) == 1:
                ones_count += 1
            else:
                zeros_count += 1
        gini_index = 1 - ((ones_count/total_records)*(ones_count/total_records)) - ((zeros_count/total_records)*(zeros_count/total_records))
        return gini_index

    def get_positives_negatives_count(self, data):
        total_records = data.shape[0]

        if total_records == 0:
            return 0.0
        prediction_value_index = data.shape[1] - 1
        ones_count = 0
        zeros_count = 0
        for record in data:
            if int(record[prediction_value_index]) == 1:
                ones_count += 1
            else:
                zeros_count += 1
        return ones_count, zeros_count

    def categorise_data(self, data, val, attribute_index):
        # divide the dataset, which satisfies the less than equal and greater than condition
        less_than_attr_value = []
        greater_than_attr_value = []

        for record in data:
            if attribute_index in self.categorical_columns_index:
                if float(record[attribute_index]) == val:
                    less_than_attr_value.append(record)
                else:
                    greater_than_attr_value.append(record)
            else:
                if float(record[attribute_index]) <= val:
                    less_than_attr_value.append(record)
                else:
                    greater_than_attr_value.append(record)

        return np.array(less_than_attr_value), np.array(greater_than_attr_value)

    def get_best_categorical_value(self, data, attribute_index, gini_index_dataset):
        attribute_values = data[:, attribute_index]  # get all values of the attribute
        attribute_values = attribute_values.astype(np.float)
        attribute_values = set(attribute_values)
        attribute_values = list(attribute_values)

        maximum_gini_gain = float("-inf")
        best_categorical_value = 0

        for val in attribute_values:
            less_than_data, great_than_data = self.categorise_data(data, val, attribute_index)
            gini_index_lesser_values = self.get_gini_index(less_than_data)
            gini_index_great_values = self.get_gini_index(great_than_data)
            total_records = data.shape[0]
            gini_children = ((less_than_data.shape[0]/data.shape[0]) * gini_index_lesser_values) + ((great_than_data.shape[0]/data.shape[0]) *  gini_index_great_values)
            gini_gain = gini_index_dataset - gini_children

            if gini_gain > maximum_gini_gain:
                maximum_gini_gain = gini_gain
                best_categorical_value = val

        return maximum_gini_gain, best_categorical_value


    def find_best_attribute(self, data, gini_index_dataset):

        total_attributes = data.shape[1]-1
        maximum_gini_gain = float("-inf")
        best_attribute_index = -1
        best_attribute_value = -1

        for i in range(0, total_attributes):
            if i not in self.attribute_indexex:
                gini_gain, categorical_value = self.get_best_categorical_value(data, i, gini_index_dataset)
                if gini_gain > maximum_gini_gain:
                    maximum_gini_gain = gini_gain
                    best_attribute_index = i
                    best_attribute_value = categorical_value

        print("Best Attribute:", best_attribute_index)
        print("Gini Value::", gini_index_dataset - maximum_gini_gain)
        return best_attribute_index, best_attribute_value

    def create_decision_tree(self, train_data):

        if train_data.shape[0] == 0 :
            print("NO Data")
            leaf_node = Node(None, None)
            leaf_node.leaf_node = True
            return leaf_node


        gini_index_dataset = self.get_gini_index(train_data)
        print("gini_index of parent: ", gini_index_dataset)

        if gini_index_dataset == 0.0:
            leaf_node = Node(None, None)
            leaf_node.leaf_node = True
            ones, zeroes = self.get_positives_negatives_count(train_data)
            if ones > zeroes:
                leaf_node.prediction = 1
            else:
                leaf_node.prediction = 0
            return leaf_node


        best_attribute_index, best_attribute_value = self.find_best_attribute(train_data, gini_index_dataset)

        if best_attribute_value == -1:
            leaf_node = Node(None, None)
            leaf_node.leaf_node = True
            ones, zeroes = self.get_positives_negatives_count(train_data)
            if ones >= zeroes:
                leaf_node.prediction = 1
            else:
                leaf_node.prediction = 0
            return leaf_node

        # create a tree node now
        # attribute_index = best_attribute_index
        # attribute_value = best_attribute_value

        # add left subtree
        # add right subtree
        root = Node(best_attribute_index, best_attribute_value)
        self.attribute_indexex.append(best_attribute_index)

        less_than_data, great_than_data = self.categorise_data(train_data, best_attribute_value, best_attribute_index)
        root.left = self.create_decision_tree(less_than_data)
        # if len(self.attribute_indexex) > 0:
        #     print("Popping 1")
        #     self.attribute_indexex.pop()
        root.right = self.create_decision_tree(great_than_data)
        # if len(self.attribute_indexex) > 0:
        #     print("Popping 2")
        #     self.attribute_indexex.pop()

        return root

    def predict_output(self, root, record):
        if root.leaf_node == True:
            return root.prediction

        attb_index = root.attribute_index

        if float(record[attb_index]) <= root.attribute_value:
            return self.predict_output(root.left, record)
        else:
            return self.predict_output(root.right, record)

    def predict_test_output(self, root, test_data):
        predictions = []
        for record in test_data:
            res = self.predict_output(root, record)
            predictions.append(res)
        return predictions

    def convert_values_to_floats(self, data, index):
        attribute_values = data[:, index]
        attrb_set = []

        for val in attribute_values:
            if val not in attrb_set:
                attrb_set.append(val)

        # attribute_values = set(attribute_values)  # got unique values
        # attribute_values = list(attribute_values)

        attribute_values = attrb_set

        string_float_map = {}
        counter = 0
        for value in attribute_values:
            string_float_map[value] = float(counter)
            print(value, "   ", counter)
            counter += 1


        # now map these values to the original data
        for record in data:
            record[index] = string_float_map[record[index]]

        return data

    # convert the string data to categorical
    def preprocess(self, data):
        # check are there any attributes which has string values
        attributes_length = data.shape[1]
        string_value_indexex = []
        for i in range(0, attributes_length):
            try:
                float(data[0][i])
            except ValueError:
                print("Not a float")
                string_value_indexex.append(i)

        # got all the index attributes which are strings, now convert them to floats
        for index in string_value_indexex:
            self.categorical_columns_index.append(index)
            data = self.convert_values_to_floats(data, index)

        return data

    def print_tree_util(self, root, space):
        if root is None:
            return

        space += self.COUNT
        self.print_tree_util(root.right, space)

        print()

        for i in range(self.COUNT, space):
            print(end=" ")

        if root.leaf_node == True:
            print(root.prediction)
        else:
            print(str(root.attribute_index) + "<=" + str(round(root.attribute_value, 2)))

        self.print_tree_util(root.left, space)



    def print_tree(self, root):
        print("######################## DECISON TREE ########################")
        self.print_tree_util(root, 0)



    def process(self, data):
        k = self.config['k_fold_validation']
        data = self.preprocess(data)
        data_split = self.divide_data(data)

        accuracy = 0.0
        precision = 0.0
        recall = 0.0
        f1_score = 0.0

        accuracy_array = []
        precision_array = []
        recall_array = []
        f1_score_array = []

        for i in range(0,k):
            test_data = data_split[i]
            train_data = None
            for j in range(0,k):
                if j != i:
                    if train_data is None:
                        train_data = data_split[j]
                    else:
                        train_data = np.append(train_data,data_split[j], axis=0)
            root = self.create_decision_tree(train_data)

            self.print_tree(root)

            predicted_values = self.predict_test_output(root, test_data)

            metric_calculator = MetricsCalculator()
            accuracy_array.append(metric_calculator.calculate_accuracy(test_data, predicted_values))
            precision = metric_calculator.calculate_precision(test_data, predicted_values)
            precision_array.append(precision)
            recall = metric_calculator.calculate_recall(test_data, predicted_values)
            recall_array.append(recall)
            f1_score = metric_calculator.calculate_F1_score(precision, recall)
            f1_score_array.append(f1_score)
            self.attribute_indexex = []

        accuracy = float(sum(accuracy_array)/k)
        precision = float(sum(precision_array)/k)
        recall = float(sum(recall_array)/k)
        f1_score = float(sum(f1_score_array) / k)

        print("K-Fold Accuracies: ", '[%s]' % ', '.join(map(str, accuracy_array)))
        print("K-Fold Precision: ", '[%s]' % ', '.join(map(str, precision_array)))
        print("K-Fold Recall: ", '[%s]' % ', '.join(map(str, recall_array)))
        print("")
        print("Accuracy: " + str(accuracy))
        print("Precision: " + str(precision))
        print("Recall: " + str(recall))
        print("F1 Score: " + str(f1_score))

        metric_calculator.plot_graph(accuracy_array, precision_array, recall_array)


        # metric_calculator = MetricsCalculator()
        # accuracy = metric_calculator.calculate_accuracy(test_data, predicted_values)
        # print("Accuracy: ", accuracy)
        # precision = metric_calculator.calculate_precision(test_data, predicted_values)
        # print("Precision: ", precision)
        # recall = metric_calculator.calculate_recall(test_data, predicted_values)
        # print("Recall: ", recall)
        # f1_score = metric_calculator.calculate_F1_score(precision, recall)
        # print("F1 Score: ", f1_score)

    def process_full(self, data):
        data = self.preprocess(data)
        train_data, test_data = self.divide_data(data)
        root = self.create_decision_tree(train_data)
        print("Here")
        self.print_tree(root)

        predicted_values = self.predict_test_output(root, test_data)

        # metric_calculator = MetricsCalculator()
        # accuracy = metric_calculator.calculate_accuracy(test_data, predicted_values)
        # print("Accuracy: ", accuracy)
        # precision = metric_calculator.calculate_precision(test_data, predicted_values)
        # print("Precision: ", precision)
        # recall = metric_calculator.calculate_recall(test_data, predicted_values)
        # print("Recall: ", recall)
        # f1_score = metric_calculator.calculate_F1_score(precision, recall)
        # print("F1 Score: ", f1_score)


def main():
    dt_classifier = DecisionTree()
    with open('dt_config.json', 'r') as f:
        config = json.load(f)
    dt_classifier.config = config
    data = dt_classifier.read_file(dt_classifier.config['input_file'])
    dt_classifier.process_full(data)

if __name__ == '__main__':
    main()