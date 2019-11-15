import sys


class MetricCalculator:

    # accuracy metric calculation
    @staticmethod
    def calculate_accuracy(test_data, predicted_labels):
        prediction_value_index = test_data.shape[1] - 1
        correct_predictions = 0

        for i in range(0, len(predicted_labels)):
            print(test_data[i][prediction_value_index] + " " + predicted_labels[i])
            if int(test_data[i][prediction_value_index]) == int(predicted_labels[i]):
                correct_predictions += 1

        accuracy = float((correct_predictions / len(predicted_labels))) * 100
        return accuracy


metric_calculator = MetricCalculator()
