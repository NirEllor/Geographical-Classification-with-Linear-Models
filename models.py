
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
import helpers

TEST_CSV = 'test.csv'
VALIDATION_CSV = 'validation.csv'
TRAIN_CSV = 'train.csv'
DATA = [TRAIN_CSV, TEST_CSV, VALIDATION_CSV]
CHOICES = [0, 2, 4, 8, 10]


torch.manual_seed(42)

class Ridge_Regression:

    def __init__(self, lamda):
        self.lamda = lamda
        self.W = None

    def fit(self, X, Y):

        """
        Fit the ridge regression model to the provided data.
        :param X: The training features.
        :param Y: The training labels.
        """

        Y = 2 * (Y - 0.5)  # Transform labels {0, 1} -> {-1, 1}

        N_train, D = X.shape

        # Add a row of ones to X for the bias term
        X_prime = np.vstack((X.T, np.ones((1, N_train))))  # Shape: (D+1, N_train)

        # Compute the covariance matrix and add regularization
        first_comp = (X_prime @ X_prime.T) / N_train + self.lamda * np.eye(D + 1)  # Shape: (D+1, D+1)

        # Compute the projection of Y onto X_prime
        second_comp = (X_prime @ Y.T) / N_train  # Shape: (D+1, 1)

        # Compute the optimal weights
        self.W = (np.linalg.inv(first_comp) @ second_comp)  # Shape: (1, D+1)

    def predict(self, X):
        """
        Predict the output for the provided data.
        :param X: The data to predict. np.ndarray of shape (N, D).
        :return: The predicted output. np.ndarray of shape (N,), of 0s and 1s.
        """

        N_test = X.shape[0]
        # Add a row of ones to X for the bias term
        X_test = np.vstack((X.T, np.ones((1, N_test))))  # Shape: (D+1, N_test)
        # Compute predictions
        predictions = self.W @ X_test  # Shape: (1, N_test)

        # Transform predictions back to {0, 1}
        predictions = (predictions + 1) / 2
        return np.where(predictions >= 0.5, 1, 0)

def load_train_data():
    train_data, _ = helpers.read_data_demo()
    X_train, y_train = train_data[:, :-1], train_data[:, -1]
    return X_train, y_train


def run_ridge_regression():
    accuracies = np.ndarray([len(DATA), len(CHOICES)])
    X_train, y_train = load_train_data()
    for file_index, file in  enumerate(DATA):
        file_name = file.split(".", 1)[0].capitalize()
        print(f"---------------- {file_name} -----------------")
        data, _ = helpers.read_data_demo(file)
        # Extract features and labels
        X, y = data[:, :-1], data[:, -1]
        for choice_index, choice in enumerate(CHOICES):

            # Initialize, train, and predict
            ridge_regression = Ridge_Regression(choice)
            ridge_regression.fit(X_train, y_train)
            y_pred = ridge_regression.predict(X)

            # Calculate accuracy
            pure_accuracy = np.mean(y_pred == y)
            accuracy = round(pure_accuracy, 6)
            accuracies[file_index, choice_index] = accuracy
            print(f"Accurac for lambda  = {choice}:", accuracy)
    # Plot results
    plot_accuracies(accuracies)

def plot_accuracies(accuracies):
    plt.figure(figsize=(10, 6))
    for file_index, file in enumerate(DATA):
        file_name = file.split(".", 1)[0].capitalize()
        plt.plot(CHOICES, accuracies[file_index], label=file_name, marker='o')
    # Customize the plot
    plt.xlabel("Lambda (Regularization Parameter)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Lambda for Different Datasets")

    # Set custom x-axis ticks
    plt.xticks(CHOICES, [f"{choice}" for choice in CHOICES])  # Format as 0.0, 0.2, etc.

    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()


class Logistic_Regression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Logistic_Regression, self).__init__()

        ########## YOUR CODE HERE ##########

        # define a linear operation.

        ####################################
        pass

    def forward(self, x):
        """

        Computes the output of the linear operator.
        :param x: The input to the linear operator.
        :return: The transformed input.
        """
        # compute the output of the linear operator

        ########## YOUR CODE HERE ##########

        # return the transformed input.
        # first perform the linear operation
        # should be a single line of code.

        ####################################

        pass

    def predict(self, x):
        """
        THIS FUNCTION IS NOT NEEDED FOR PYTORCH. JUST FOR OUR VISUALIZATION
        """
        x = torch.from_numpy(x).float().to(self.linear.weight.data.device)
        x = self.forward(x)
        x = nn.functional.softmax(x, dim=1)
        x = x.detach().cpu().numpy()
        x = np.argmax(x, axis=1)
        return x

if __name__ == '__main__':
    run_ridge_regression()