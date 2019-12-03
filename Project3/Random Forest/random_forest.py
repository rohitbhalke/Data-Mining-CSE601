import numpy as np
import sys
from Node import Node as Node
from PerformanceMetrics import MetricsCalculator
from Decision_Tree.decision_tree import DecisionTree
import json
import math
import random

class RandomForest(DecisionTree):

    # Overridden method
    def find_best_attribute(self, data, gini_index_dataset):

        total_attributes = data.shape[1]-1
        maximum_gini_gain = float("-inf")
        best_attribute_index = -1
        best_attribute_value = -1


        attribute_set = []
        # first get all attributes in a list
        for i in range(0, total_attributes):
            attribute_set.append(i)

        filtered_features = []

        # choose sqrt(no_of_features as a subset of attributes)
        no_of_filtered_features = self.config['features_for_split']

        # find random m attributes/features for each node split
        while len(filtered_features) != no_of_filtered_features:
            random_num = random.randint(0, total_attributes)
            filtered_features.append(random_num)
            filtered_features = set(filtered_features)
            filtered_features = list(filtered_features)

        # find best attribute among the m attribute
        for i in filtered_features:
            gini_gain, categorical_value = self.get_best_categorical_value(data, i, gini_index_dataset)
            if gini_gain > maximum_gini_gain:
                maximum_gini_gain = gini_gain
                best_attribute_index = i
                best_attribute_value = categorical_value

        print("Best Attribute:", best_attribute_index)
        return best_attribute_index, best_attribute_value

    # Overridden method
    def create_decision_tree(self, train_data):

        if train_data.shape[0] == 0 :
            print("NO Data")
            leaf_node = Node(None, None)
            leaf_node.leaf_node = True
            return leaf_node


        gini_index_dataset = self.get_gini_index(train_data)
        print("gini_index : ", gini_index_dataset)

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

        less_than_data, great_than_data = self.categorise_data(train_data, best_attribute_value, best_attribute_index)
        root.left = self.create_decision_tree(less_than_data)

        root.right = self.create_decision_tree(great_than_data)
        return root

    def bagging_ensemble(self, random_forest_predictions):
        print("Start Ensembler")
        bagging_output = []

        for i in range(0, len(random_forest_predictions[0])):
            no_of_ones = 0
            no_of_zeros = 0
            for prediction in random_forest_predictions:
                if prediction[i] == 1:
                    no_of_ones += 1
                if prediction[i] == 0:
                    no_of_zeros += 1

            if no_of_ones > no_of_zeros:
                bagging_output.append(1)
            else:
                bagging_output.append(0)

        return bagging_output

    # Overridden method
    def process(self, data):
        data = self.preprocess(data)
        train_data, test_data = self.divide_data(data)

        no_of_decision_trees = self.config['n_estimators']
        roots = []

        # create decision tress with diff nodes
        for i in range(0, no_of_decision_trees):
            root = self.create_decision_tree(train_data)
            print("Constructed Decision Tree Number: ", i+1)
            roots.append(root)

        random_forest_predictions = []

        # find prediction output values for each decision tree
        for i in range(0, no_of_decision_trees):
            predicted_values = self.predict_test_output(roots[i], test_data)
            random_forest_predictions.append(predicted_values)

        # Implement ensembler technique (bagging) for finding the final predictions
        random_forest_predictions = self.bagging_ensemble(random_forest_predictions)

        metric_calculator = MetricsCalculator()
        accuracy = metric_calculator.calculate_accuracy(test_data, random_forest_predictions)
        print("Accuracy: ", accuracy)
        precision = metric_calculator.calculate_precision(test_data, random_forest_predictions)
        print("Precision: ", precision)
        recall = metric_calculator.calculate_recall(test_data, random_forest_predictions)
        print("Recall: ", recall)
        f1_score = metric_calculator.calculate_F1_score(precision, recall)
        print("F1 Score: ", f1_score)

    def divide_data(self, data):
        k = self.config['k_fold_validation']
        no_of_records = int(data.shape[0] / k)
        data_split = {}
        for i in range(0,k-1):
            start = i*no_of_records
            end = start + no_of_records
            data_split[i] = data[start:end]
        data_split[i+1] = data[end:]
        return data_split

    def process1(self, data):
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

        no_of_decision_trees = self.config['n_estimators']


        # create decision tress with diff nodes
        for l in range(0, no_of_decision_trees):
            roots = []
            for i in range(0, k):
                test_data = data_split[i]
                train_data = None
                for j in range(0, k):
                    if j != i:
                        if train_data is None:
                            train_data = data_split[j]
                        else:
                            train_data = np.append(train_data, data_split[j], axis=0)

                root = self.create_decision_tree(train_data)
                print("Constructed Decision Tree Number: ", i+1)
                roots.append(root)

            random_forest_predictions = []
            # find prediction output values for each decision tree
            for r in range(0, len(roots)):
                predicted_values = self.predict_test_output(roots[r], test_data)
                random_forest_predictions.append(predicted_values)

            # Implement ensembler technique (bagging) for finding the final predictions
            random_forest_predictions = self.bagging_ensemble(random_forest_predictions)

            metric_calculator = MetricsCalculator()
            accuracy = metric_calculator.calculate_accuracy(test_data, random_forest_predictions)
            accuracy_array.append(accuracy)
            precision = metric_calculator.calculate_precision(test_data, random_forest_predictions)
            precision_array.append(precision)
            recall = metric_calculator.calculate_recall(test_data, random_forest_predictions)
            recall_array.append(recall)
            f1_score = metric_calculator.calculate_F1_score(precision, recall)
            f1_score_array.append(f1_score)


        accuracy = float(sum(accuracy_array) / no_of_decision_trees)
        precision = float(sum(precision_array) / no_of_decision_trees)
        recall = float(sum(recall_array) / no_of_decision_trees)
        f1_score = float(sum(f1_score_array) / no_of_decision_trees)

        print("K-Fold Accuracies: ", '[%s]' % ', '.join(map(str, accuracy_array)))
        print("K-Fold Precision: ", '[%s]' % ', '.join(map(str, precision_array)))
        print("K-Fold Recall: ", '[%s]' % ', '.join(map(str, recall_array)))
        print("")
        print("Accuracy: " + str(accuracy))
        print("Precision: " + str(precision))
        print("Recall: " + str(recall))
        print("F1 Score: " + str(f1_score))

        metric_calculator.plot_graph(accuracy_array, precision_array, recall_array)


def main():
    rf_classifier = RandomForest()
    with open('rf_config.json', 'r') as f:
        config = json.load(f)
    rf_classifier.config = config
    data = rf_classifier.read_file(rf_classifier.config['input_file'])
    rf_classifier.process1(data)

if __name__ == '__main__':
    main()