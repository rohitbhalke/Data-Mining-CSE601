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
        no_of_filtered_features = int(math.sqrt(total_attributes))

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


def main():
    rf_classifier = RandomForest()
    with open('rf_config.json', 'r') as f:
        config = json.load(f)
    rf_classifier.config = config
    data = rf_classifier.read_file(rf_classifier.config['input_file'])
    rf_classifier.process(data)

if __name__ == '__main__':
    main()