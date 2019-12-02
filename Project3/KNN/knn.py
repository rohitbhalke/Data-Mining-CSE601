import json
import numpy as np
from PerformanceMetrics import MetricsCalculator


class KNNClassifier:

    def __init__(self):
        self.config = None
        self.k_fold = 0
        self.k = 0

    def read_file(self, path):
        data = []
        with open(path) as fp:
            line = fp.readline()
            pt = 1
            while line:
                value = line.replace("\n", "").split("\t")
                value.insert(0, pt)
                pt += 1
                data.append(value)
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

    def euclidian_distance(self, x1, x2):  # it is used for calculating euclidean distance
        distance = 0
        for x in range(len(x1)):
            distance += np.square(float(x1[x]) - float(x2[x]))
        return np.sqrt(distance)

    def classification_demo(self):
        predicted_labels = []
        test_data = self.read_file(self.config['test_data_file'])
        train_data = self.read_file(self.config['train_data_file'])

        predicted_labels = self.classify(test_data, train_data)

        metric_calculator = MetricsCalculator()
        accuracy = metric_calculator.calculate_accuracy(np.array(test_data), predicted_labels)
        precision = metric_calculator.calculate_precision(np.array(test_data), predicted_labels)
        recall = metric_calculator.calculate_recall(np.array(test_data), predicted_labels)

        f1_score = metric_calculator.calculate_F1_score(precision, recall)

        print("Accuracy: " + str(accuracy))
        print("Precision: " + str(precision))
        print("Recall: " + str(recall))
        print("F1 Score: " + str(f1_score))

    def process_and_classify_data(self, data):
        rows = len(data)
        cols = len(data[0])
        data = self.preprocess(data)
        data_split = self.divide_data(data, rows, cols)
        test_data = []
        accuracy = 0.0
        precision = 0.0
        recall = 0.0
        f1_score = 0.0

        accuracy_array = []
        precision_array = []
        recall_array = []
        f1_score_array = []

        predicted_labels = []
        metric_calculator = MetricsCalculator()
        for i in range(0, self.k_fold):
            test_data = data_split[i]
            train_data = []
            for j in range(0, self.k_fold):
                if j != i:
                    for d in data_split[j]:
                        train_data.append(d)

            predicted_labels = self.classify(test_data, train_data)

            accuracy_array.append(metric_calculator.calculate_accuracy(test_data, predicted_labels))
            precision = metric_calculator.calculate_precision(test_data, predicted_labels)
            precision_array.append(precision)
            recall = metric_calculator.calculate_recall(test_data, predicted_labels)
            recall_array.append(recall)
            f1_score = metric_calculator.calculate_F1_score(precision, recall)
            f1_score_array.append(f1_score)

            accuracy += metric_calculator.calculate_accuracy(np.array(test_data), predicted_labels)
            precision += metric_calculator.calculate_precision(np.array(test_data), predicted_labels)
            recall += metric_calculator.calculate_recall(np.array(test_data), predicted_labels)

        accuracy = float(sum(accuracy_array) / self.k_fold)
        precision = float(sum(precision_array) / self.k_fold)
        recall = float(sum(recall_array) / self.k_fold)
        f1_score = float(sum(f1_score_array) / self.k_fold)

        print("K-Fold Accuracies: ", '[%s]' % ', '.join(map(str, accuracy_array)))
        print("K-Fold Precision: ", '[%s]' % ', '.join(map(str, precision_array)))
        print("K-Fold Recall: ", '[%s]' % ', '.join(map(str, recall_array)))
        print("")
        print("Accuracy: " + str(accuracy))
        print("Precision: " + str(precision))
        print("Recall: " + str(recall))
        print("F1 Score: " + str(f1_score))

        metric_calculator.plot_graph(accuracy_array, precision_array, recall_array)

    def divide_data(self, data, rows, cols):
        no_of_records = int(rows / self.k_fold)
        data_split = {}
        for i in range(0, self.k_fold - 1):
            start = i * no_of_records
            end = start + no_of_records
            data_split[i] = data[start:end]
        data_split[i + 1] = data[end:]
        return data_split

    def classify(self, test_data, train_data):
        predicted_class_labels = []
        for i in range(len(test_data)):
            class0 = 0
            class1 = 0
            neighbors = {}
            for j in range(len(train_data)):
                distance = self.euclidian_distance(test_data[i][1:len(test_data[i])-1], train_data[j][1:len(train_data[j])-1])
                pts = [distance, train_data[j][0], int(train_data[j][len(train_data[j])-1])]
                neighbors[j] = pts

            neighbors = sorted(neighbors.items(), key=lambda x: x[1])
            for cnt in range(self.k):
                val = neighbors[cnt][1]
                if val[2] == 0:
                    class0 += (1/(val[0]))
                    # class0 += 1
                else:
                    class1 += (1/(val[0]))
                    # class1 += 1
            if class0 > class1:
                predicted_class_labels.append(0)
            else:
                predicted_class_labels.append(1)
        return predicted_class_labels


def main():
    knn_classifier = KNNClassifier()
    with open('knn_config.json', 'r') as f:
        config = json.load(f)
    knn_classifier.config = config
    knn_classifier.k = knn_classifier.config['k']
    knn_classifier.k_fold = knn_classifier.config['k_fold_validation']
    knn_classifier.classification_demo()
    data = knn_classifier.read_file(knn_classifier.config['input_file'])
    knn_classifier.process_and_classify_data(data)


if __name__ == '__main__':
    main()
