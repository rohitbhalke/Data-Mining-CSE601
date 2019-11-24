import json
import numpy as np
from PerformanceMetrics import MetricsCalculator
from scipy.spatial import distance as dist


class KNNClassifier:

    def __init__(self):
        self.config = None
        self.k = 0
        self.train_attrs = []
        self.train_class_labels = []
        self.test_attrs = []
        self.test_class_labels = []
        self.predicted_class_labels = []

    def read_file(self, path):
        data = []
        with open(path) as fp:
            line = fp.readline()
            while line:
                value = line.replace("\n", "").split("\t")
                data.append(value)
                line = fp.readline()
        return data

    def process_data(self, data_points):
        train_data_percent = self.config['training_percentage']
        no_of_records = int(len(data_points) * (train_data_percent / 100))
        print(no_of_records)

        for i in range(len(data_points)):
            feature_list = data_points[i]
            if i < no_of_records:
                self.train_class_labels.append(int(feature_list[len(feature_list) - 1]))
            else:
                self.test_class_labels.append(int(feature_list[len(feature_list) - 1]))
            pts = []
            for j in range(len(feature_list[0: len(feature_list) - 1])):
                pts.append(float(feature_list[j]))

            if i < no_of_records:
                self.train_attrs.append(pts)
            else:
                self.test_attrs.append(pts)

    # convert the string data to categorical

    def pre_process(self, data):
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

    # pre processing input features

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

    def divide_data(self, data):
        train_data_percent = self.config['training_percentage']
        no_of_records = int(data.shape[0] * (train_data_percent / 100))
        train_data = data[0:no_of_records]
        test_data = data[no_of_records:]
        return train_data, test_data

    def classify(self):
        for i in range(len(self.test_attrs)):
            class0 = 0
            class1 = 0
            neighbors = {}
            for j in range(len(self.train_attrs)):
                neighbors[j] = distance = dist.euclidean(self.train_attrs[j], self.test_attrs[i])

            neighbors = sorted(neighbors.items(), key=lambda x: x[1])
            for cnt in range(0, self.k):
                val = int(neighbors[cnt][0])
                if self.train_class_labels[val] == 0:
                    class0 = class0 + 1
                else:
                    class1 = class1 + 1
            if class0 > class1:
                self.predicted_class_labels.append(0)
            else:
                self.predicted_class_labels.append(1)
        print(len(self.predicted_class_labels))

    def calculate_measures(self, data):
        data = self.pre_process(np.array(data))
        train_data_split, test_data_split = self.divide_data(data)
        metric_calculator = MetricsCalculator()
        accuracy = metric_calculator.calculate_accuracy(test_data_split, self.predicted_class_labels)
        print("Accuracy: " + str(accuracy))
        precision = metric_calculator.calculate_precision(test_data_split, self.predicted_class_labels)
        print("Precision: ", precision)
        recall = metric_calculator.calculate_recall(test_data_split, self.predicted_class_labels)
        print("Recall: ", recall)
        f1_score = metric_calculator.calculate_F1_score(precision, recall)
        print("F1 Score: ", f1_score)


def main():
    knn_classifier = KNNClassifier()
    with open('knn_config.json', 'r') as f:
        config = json.load(f)
    knn_classifier.config = config
    knn_classifier.k = knn_classifier.config['k']
    data = knn_classifier.read_file(knn_classifier.config['input_file'])
    knn_classifier.process_data(data)
    knn_classifier.classify()
    knn_classifier.calculate_measures(data)


if __name__ == '__main__':
    main()
