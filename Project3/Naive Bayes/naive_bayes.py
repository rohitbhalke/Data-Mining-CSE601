import numpy as np
import sys
import math
from Project3.PerformanceMetrics import MetricsCalculator
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
        string_value_indexes = []
        for i in range(0, attributes_length):
            try:
                float(data[0][i])
            except ValueError:
                string_value_indexes.append(i)

        # got all the index attributes which are strings, now convert them to floats
        for index in string_value_indexes:
            data = self.convert_values_to_floats(data, index)
        return data, string_value_indexes

    def divide_data(self,data):
        k = self.config['k_fold_validation']
        no_of_records = int(data.shape[0] / k)
        data_split = {}
        for i in range(0,k-1):
            start = i*no_of_records
            end = start + no_of_records
            data_split[i] = data[start:end]
        data_split[i+1] = data[end:]
        return data_split

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
    def calculate_mu_var_by_class(self, data, string_value_indexes):
        data_summary = {}
        for class_value, rows in data.items():
            rows = np.asarray(rows)
            summaries = {}
            for i in range(0,len(rows[0])):
                column = rows[:, i:i + 1]
                column = list(column)
                if i not in string_value_indexes:
                    summaries[i] = ([(self.mean(column), self.stdev(column), len(column))])
                else:
                    summaries[i] = ([(" ", " ", len(column))])
            data_summary[class_value] = summaries
        return data_summary


    def calculate_poeterior_probability(self, data, string_value_indexes):
        data_summary = {}
        for i in string_value_indexes:
            summary_index = {}
            for class_value, rows in data.items():
                summaries = {}
                total_rows = len(rows)
                rows = np.asarray(rows)
                column = rows[:,i:i+1]
                column = column.flatten()
                keys, frequencies = np.unique(column, return_counts=True)
                for j in range(0,len(keys)):
                    prob = float(frequencies[j] / total_rows)
                    if prob == 0.0:
                        summaries = self.calculate_laplacian_correct_probability(rows, keys, frequencies)
                        break
                    summaries[keys[j]] = prob
                summary_index[class_value] = summaries
            data_summary[i] = summary_index
        return data_summary

    def calculate_laplacian_correct_probability(self, rows, keys, frequencies):
        total_rows = len(rows)
        total_rows += len(keys)
        frequencies += 1
        summaries = {}
        for i in range(0,len(keys)):
            prob = float(frequencies[i] / total_rows)
            summaries[keys[i]] = prob
        return summaries


    # calculate gaussian probability as the features are continuous
    def calculate_gaussian_probability(self, x, mean, covar):
        x = x.astype(np.float)
        exponent = math.exp(-((x - mean) ** 2 / (2 * covar ** 2)))
        return (1 / (math.sqrt(2 * math.pi) * covar)) * exponent

    # for each test_data vector(dx1), calculate P(X|H)*P(H)
    def calculate_probabilities_by_class(self, data_summary_continuous, data_summary_categorical, data_vector, string_value_indexes):
        data_vector = data_vector[:len(data_vector)-1]
        total_rows = 0
        for label in data_summary_continuous:
            total_rows += data_summary_continuous.get(label)[0][0][2]
        # total_rows = sum([data_summary[label][0][2] for label in data_summary])
        probabilities = dict()
        for class_value, class_summaries in data_summary_continuous.items():
            probabilities[class_value] = data_summary_continuous.get(class_value)[0][0][2] / float(total_rows)
            # probabilities[class_value] = data_summary[class_value][0][2] / float(total_rows)
            for i in range(0,len(data_vector)):
                if i not in string_value_indexes:
                    mean, stdev, _ = class_summaries[i][0]
                    probabilities[class_value] *= self.calculate_gaussian_probability(data_vector[i], mean, stdev)
                else:
                    prob = data_summary_categorical.get(i).get(class_value).get(data_vector[i])
                    probabilities[class_value] *= prob
        print("----Probability---")
        print(data_vector)
        for class_value, prob in probabilities.items():
            print("Class: " + str(class_value) + " probability: " + str(prob))
        return probabilities

    # predict label for given data sample with highest P(Xi|H)*P(H)
    def predict(self, data, data_summary_continuous, data_summary_categorical, string_value_indexes):
        predictions = []
        for item in data:
            probabilities = self.calculate_probabilities_by_class(data_summary_continuous, data_summary_categorical, item, string_value_indexes)
            best_label, best_prob = None, -1
            for class_value, probability in probabilities.items():
                if best_label is None or probability > best_prob:
                    best_prob = probability
                    best_label = class_value
            predictions.append(int(best_label))
        return predictions

    def classify(self, train_data, test_data, string_value_indexes):
        train_data_split_by_class = self.split_by_class(train_data)
        data_summary_continuous_features = self.calculate_mu_var_by_class(train_data_split_by_class, string_value_indexes)
        data_summary_categorical_features = self.calculate_poeterior_probability(train_data_split_by_class, string_value_indexes)
        predictions = self.predict(test_data,data_summary_continuous_features,data_summary_categorical_features , string_value_indexes)
        return predictions

    def process(self, data):
        k = self.config['k_fold_validation']
        data, string_value_indexes = self.preprocess(data)
        data_split = self.divide_data(data)
        train_data = None
        test_data = None
        accuracy = 0.0
        precision = 0.0
        recall = 0.0
        f1_score = 0.0
        for i in range(0,k):
            test_data = data_split[i]
            train_data = None
            for j in range(0,k):
                if j != i:
                    if train_data is None:
                        train_data = data_split[j]
                    else:
                        train_data = np.append(train_data,data_split[j], axis=0)
            predicted_labels = self.classify(train_data,test_data,string_value_indexes)

            metric_calculator = MetricsCalculator()
            accuracy += metric_calculator.calculate_accuracy(test_data, predicted_labels)
            precision += metric_calculator.calculate_precision(test_data, predicted_labels)
            recall += metric_calculator.calculate_recall(test_data, predicted_labels)

        accuracy = float(accuracy/k)
        precision = float(precision/k)
        recall = float(recall/k)
        f1_score = metric_calculator.calculate_F1_score(precision, recall)

        print("Accuracy: " + str(accuracy))
        print("Precision: " + str(precision))
        print("Recall: " + str(recall))
        print("F1 Score: " + str(f1_score))

    def process_demo(self, data):
        train_samples = self.config['demo_train_samples']
        data, string_value_indexes = self.preprocess(data)
        train_data = data[0:train_samples, :]
        test_data = data[train_samples:, :]
        predicted_labels = self.classify(train_data, test_data, string_value_indexes)

        print("---Predicted Labels---")
        for item in predicted_labels:
            print(item)


def main():
    nb_classifier = NaiveBayesClassifier()
    with open('nb_config.json', 'r') as f:
        config = json.load(f)
    nb_classifier.config = config
    data = nb_classifier.read_file(nb_classifier.config['input_file'])
    # nb_classifier.process(data)
    nb_classifier.process_demo(data)
if __name__ == '__main__':
    main()