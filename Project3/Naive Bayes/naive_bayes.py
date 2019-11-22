import numpy as np
import sys
import math
from PerformanceMetrics import MetricsCalculator
import json


class NaiveBayesClassifier:

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

    # preprocessing input features
    def convert_values_to_floats(self, data, index):
        attribute_values = data[:, index]
        attribute_values = set(attribute_values)  # got unique values
        attribute_values = list(attribute_values)

        string_float_map = {}
        counter = 0
        for value in attribute_values:
            string_float_map[value] = float(counter)
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
                string_value_indexex.append(i)

        # got all the index attributes which are strings, now convert them to floats
        for index in string_value_indexex:
            data = self.convert_values_to_floats(data, index)

        return data

    def divide_data(self, data):
        train_data_percent = self.config['training_percentage']
        no_of_records = int(data.shape[0] * (train_data_percent / 100))
        train_data = data[0:no_of_records]
        test_data = data[no_of_records:]
        return train_data, test_data

    # first split data by class, dict to store this data
    def split_by_class(self, data):
        separated_data = dict()
        label_index = data.shape[1]
        for i in range(len(data)):
            vector = data[i]
            class_value = vector[label_index-1]
            vector = vector[0:label_index-1]
            if (class_value not in separated_data):
                separated_data[class_value] = list()
            separated_data[class_value].append(vector)
        return separated_data

    # calculate mean of feature
    def mean(self, numbers):
        numbers = np.asarray(numbers).astype(np.float)
        return sum(numbers) / float(len(numbers))

    # calculate standard deviation of feature
    def stdev(self, numbers):
        numbers = np.asarray(numbers).astype(np.float)
        avg = self.mean(numbers)
        variance = sum([(x - avg) ** 2 for x in numbers]) / float(len(numbers) - 1)
        return math.sqrt(variance)

    # summarize train_data to get the gaussian hyperparameters (mu, sigma) foe each class feature-wise
    def calculate_mu_var_by_class(self,data):
        data_summary = {}
        for class_value, rows in data.items():
            summaries = []
            for column in zip(*rows):
                summaries.append([(self.mean(column), self.stdev(column), len(column))])
            data_summary[class_value] = summaries
        return data_summary

    # calculate gaussian probability as the features are continuous
    def calculate_probability(self, x, mean, covar):
        x = x.astype(np.float)
        exponent = math.exp(-((x - mean) ** 2 / (2 * covar ** 2)))
        return (1 / (math.sqrt(2 * math.pi) * covar)) * exponent

    # for each test_data vector(dx1), calculate P(X|H)*P(H)
    def calculate_probabilities_by_class(self, data_summary, data_vector):
        data_vector = data_vector[:len(data_vector)-1]
        total_rows = 0
        for label in data_summary:
            total_rows += data_summary.get(label)[0][0][2]
        # total_rows = sum([data_summary[label][0][2] for label in data_summary])
        probabilities = dict()
        for class_value, class_summaries in data_summary.items():
            probabilities[class_value] = data_summary.get(class_value)[0][0][2] / float(total_rows)
            # probabilities[class_value] = data_summary[class_value][0][2] / float(total_rows)
            for i in range(len(class_summaries)):
                mean, stdev, _ = class_summaries[i][0]
                probabilities[class_value] *= self.calculate_probability(data_vector[i], mean, stdev)
        return probabilities

    # predict label for given data sample with highest P(Xi|H)*P(H)
    def predict(self, data, data_summary):
        predictions = []
        for item in data:
            probabilities = self.calculate_probabilities_by_class(data_summary, item)
            best_label, best_prob = None, -1
            for class_value, probability in probabilities.items():
                if best_label is None or probability > best_prob:
                    best_prob = probability
                    best_label = class_value
            predictions.append(int(best_label))
        return predictions

    def classify(self, train_data, test_data):
        train_data_split_by_class = self.split_by_class(train_data)
        data_summary = self.calculate_mu_var_by_class(train_data_split_by_class)
        predictions = self.predict(test_data,data_summary)
        return predictions

    def process(self, data):
        data = self.preprocess(data)
        train_data_split, test_data_split = self.divide_data(data)
        predicted_labels = self.classify(train_data_split,test_data_split)
        metric_calculator = MetricsCalculator()
        accuracy = metric_calculator.calculate_accuracy(test_data_split, predicted_labels)
        print("Accuracy: " + str(accuracy))
        precision = metric_calculator.calculate_precision(test_data_split, predicted_labels)
        print("Precision: ", precision)
        recall = metric_calculator.calculate_recall(test_data_split, predicted_labels)
        print("Recall: ", recall)
        f1_score = metric_calculator.calculate_F1_score(precision, recall)
        print("F1 Score: ", f1_score)

def main():
    nb_classifier = NaiveBayesClassifier()
    with open('nb_config.json', 'r') as f:
        config = json.load(f)
    nb_classifier.config = config
    data = nb_classifier.read_file(nb_classifier.config['input_file'])
    nb_classifier.process(data)

if __name__ == '__main__':
    main()