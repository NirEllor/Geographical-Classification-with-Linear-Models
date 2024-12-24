import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
import helpers
from sklearn.tree import DecisionTreeClassifier


GRADIENT_DESCENT_ITERATIONS = 1000

GRADIENT_DESCENT_LEARNING_RATE = 0.1

# Constants for CSV file paths
TRAIN_CSV = 'train.csv'
VALIDATION_CSV = 'validation.csv'
TEST_CSV = 'test.csv'

TRAIN_MULTI = 'train_multiclass.csv'
VALIDATION_MULTI = 'validation_multiclass.csv'
TEST_MULTI = 'test_multiclass.csv'

# Hyperparameters and data configurations

RIDGE_LEARNING_RATES = [0, 2, 4, 8, 10]
BINARY_LOGISTIC_LEARNING_RATES = [0.1, 0.01, 0.001]
BINARY_DATA = [TRAIN_CSV, VALIDATION_CSV, TEST_CSV]
MULTI_LOGISTIC_LEARNING_RATES = [0.01, 0.001, 0.0003]
MULTI_DATA = [TRAIN_MULTI, VALIDATION_MULTI,  TEST_MULTI]

# Seed settings for reproducibility

torch.manual_seed(42)
np.random.seed(42)

# Configure default device and settings for Torch
torch.set_printoptions(5)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Ridge_Regression:
    """
    Implements Ridge Regression for binary classification.
    """
    def __init__(self, lamda):
        self.lamda = lamda # Regularization parameter
        self.W = None # Placeholder for weights

    def fit(self, X, Y):

        """
        Fit the ridge regression model to the provided data.
        :param X: The training features.
        :param Y: The training labels.
        """

        Y = 2 * (Y - 0.5)  # Transform labels {0, 1} -> {-1, 1}

        N_train, D = X.shape

        # Add a bias term to X
        X_prime = np.vstack((X.T, np.ones((1, N_train))))  # Shape: (D+1, N_train)

        # Compute covariance matrix with regularization
        first_comp = (X_prime @ X_prime.T) / N_train + self.lamda * np.eye(D + 1)  # Shape: (D+1, D+1)

        # Compute projection of Y onto X_prime
        second_comp = (X_prime @ Y.T) / N_train  # Shape: (D+1, 1)

        # Calculate optimal weights
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


# Helper functions for data loading and processing
def load_train_data():
    train_data, _ = helpers.read_data_demo()
    X_train, y_train = train_data[:, :-1], train_data[:, -1]
    return X_train, y_train

# Main function for running Ridge Regression
def run_ridge_regression():
    accuracies = np.ndarray([len(BINARY_DATA), len(RIDGE_LEARNING_RATES)])
    X_train, y_train = load_train_data()

    for file_index, file in enumerate(BINARY_DATA):
        process_file(file, file_index, X_train, y_train, accuracies)

    # Plot results
    plot_accuracies(accuracies)

    # Analyze validation results
    validation_sublist = accuracies[1]
    test_sublist = accuracies[2]
    max_index, min_index = analyze_validation_results(validation_sublist)
    best_test = accuracies[2][max_index]
    for index, learning_rate in enumerate(RIDGE_LEARNING_RATES):
        print(f"Validation accuracy with lambda = {learning_rate} is {validation_sublist[index]:.6f}")
        print(f"Test accuracy with lambda = {learning_rate} is {test_sublist[index]:.6f}")
    print(f"Best validation accuracy is {validation_sublist[max_index]:.6f}")
    print(f"Test accuracy according to the best validation accuracy: {best_test:.6f}")

    max_lambda, min_lambda = RIDGE_LEARNING_RATES[max_index], RIDGE_LEARNING_RATES[min_index]
    print(f"Best lambda is {max_lambda}")

    # Train and plot with best and worst lambdas
    train_and_plot(max_lambda, min_lambda)

def generate_X_y_from_csv(file):
    """
    Load features and labels from a CSV file.
    :param file: Path to the CSV file.
    :return: Tuple (X, y) where X are features and y are labels.
    """
    data, _ = helpers.read_data_demo(file)

    # Extract features and labels
    X, y = data[:, :-1], data[:, -1]
    return X, y

def process_file(file, file_index, X_train, y_train, accuracies):
    """
    Process a single file and evaluate Ridge Regression for different lambdas.
    :param file: File path for dataset.
    :param file_index: Index of the file in BINARY_DATA.
    :param X_train: Training features.
    :param y_train: Training labels.
    :param accuracies: Matrix to store accuracies.
    """
    X, y = generate_X_y_from_csv(file)

    for choice_index, choice in enumerate(RIDGE_LEARNING_RATES):
        accuracy = train_and_evaluate_ridge(choice, X_train, y_train, X, y)
        accuracies[file_index, choice_index] = accuracy
        # print(f"Accuracy for lambda = {choice}: {accuracy}")

def train_and_evaluate_ridge(lambda_value, X_train, y_train, X, y):
    """
    Train Ridge Regression for a specific lambda and evaluate its accuracy.
    :param lambda_value: Regularization parameter.
    :param X_train: Training features.
    :param y_train: Training labels.
    :param X: Validation or test features.
    :param y: Validation or test labels.
    :return: Accuracy of the model.
    """
    ridge_regression = Ridge_Regression(lambda_value)
    ridge_regression.fit(X_train, y_train)
    y_pred = ridge_regression.predict(X)

    # Calculate accuracy
    pure_accuracy = np.mean(y_pred == y)
    return round(pure_accuracy, 6)

def analyze_validation_results(validation_sublist):
    """
    Analyze the validation results to find the best and worst lambda indices.
    :param validation_sublist: Array of validation accuracies.
    :return: Indices of the maximum and minimum accuracies.
    """
    max_index = np.argmax(validation_sublist)
    min_index = np.argmin(validation_sublist)
    return max_index, min_index

def train_and_plot(max_lambda, min_lambda):
    """
    Train Ridge Regression models using the best and worst lambdas and plot decision boundaries.
    :param max_lambda: Best lambda value.
    :param min_lambda: Worst lambda value.
    """
    ridge_regression_max = Ridge_Regression(max_lambda)
    ridge_regression_min = Ridge_Regression(min_lambda)

    X, y = generate_X_y_from_csv(TEST_CSV)

    ridge_regression_max.fit(X, y)
    ridge_regression_min.fit(X, y)

    # Plot decision boundaries
    helpers.plot_decision_boundaries(ridge_regression_max, X, y,
                             f"Decision Boundaries Using best lambda = {max_lambda}")
    helpers.plot_decision_boundaries(ridge_regression_min, X, y,
                             f"Decision Boundaries Using worst lambda = {min_lambda}")

def plot_accuracies(accuracies):
    """
    Plot accuracies of Ridge Regression for different lambdas across datasets.
    :param accuracies: Matrix of accuracies for different datasets and lambdas.
    """
    plt.figure(figsize=(10, 6))
    for file_index, file in enumerate(BINARY_DATA):
        file_name = file.split(".", 1)[0].capitalize()
        plt.plot(RIDGE_LEARNING_RATES, accuracies[file_index], label=file_name, marker='o')
    plt.xlabel("Lambda")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Lambda for Different Datasets")
    plt.xticks(RIDGE_LEARNING_RATES, [f"{choice}" for choice in RIDGE_LEARNING_RATES])  # Format as 0.0, 0.2, etc.
    plt.legend()
    plt.grid(True)
    plt.show()


def gradient_descent():
    """
    Perform gradient descent to optimize a simple quadratic function and visualize the trajectory.
    """
    # Define the gradient of the function
    def gradient_f(X, Y):
        df_dx = 2 * (X - 3)
        df_dy = 2 * (Y - 5)
        return np.array([df_dx, df_dy])

    # Parameters
    learning_rate = GRADIENT_DESCENT_LEARNING_RATE
    iterations = GRADIENT_DESCENT_ITERATIONS

    # Initialize the vector (x, y)
    x, y = 0, 0
    trajectory = [(x, y)]

    # Perform gradient descent
    for _ in range(iterations):
        grad = gradient_f(x, y)
        x -= learning_rate * grad[0]
        y -= learning_rate * grad[1]
        trajectory.append((x, y))

    # Extract x and y values for plotting
    x_vals, y_vals = zip(*trajectory)

    # Plot trajectory
    plt.figure(figsize=(8, 6))
    plt.scatter(x_vals, y_vals, c=range(iterations + 1), cmap='viridis', s=10)
    plt.colorbar(label='Iteration')
    plt.title("Gradient Descent Trajectory")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid()
    # Annotate the final point
    final_point = (float(round(x_vals[-1], 2)), float(round(y_vals[-1], 2)))
    plt.text(final_point[0] - 1, final_point[1], f"Optimal Solution: {final_point}", fontsize=15, color='blue',
             ha='center')
    plt.show()

    # Print the final point
    final_point = (x_vals[-1], y_vals[-1])
    print(f"Final point is {final_point[0]:.6f}, {final_point[1]:.6f}")




class Logistic_Regression(nn.Module):
    """
    A simple Logistic Regression model using PyTorch.
    """
    def __init__(self, input_dim, output_dim):
        super(Logistic_Regression, self).__init__()


        # define a linear operation.
        self.linear = nn.Linear(in_features=input_dim, out_features=output_dim)
        # One output for binary classification


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
        return self.linear(x)

    def predict(self, x):
        """
        THIS FUNCTION IS NOT NEEDED FOR PYTORCH. JUST FOR OUR VISUALIZATION
        Generate predictions using the trained model.
        :param x: Input data as a NumPy array.
        :return: Predicted class labels.
        """
        x = torch.from_numpy(x).float().to(self.linear.weight.data.device)
        x = self.forward(x)
        x = nn.functional.softmax(x, dim=1)
        x = x.detach().cpu().numpy()
        x = np.argmax(x, axis=1)
        return x


class Dataset(torch.utils.data.Dataset):
    """
    Custom Dataset class for loading data for PyTorch.
    """

    def __init__(self, file_name):
        # Read data using the helper function
        data, _ = helpers.read_data_demo(file_name)

        # Separate features and labels
        features = data[:, :-1]
        labels = data[:, -1]

        # Convert to PyTorch tensors
        self.data = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        """
        Return the size of the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieve a single data point and its label.
        :param idx: Index of the data point.
        :return: Tuple (data, label).
        """
        return self.data[idx], self.labels[idx]

def generate_loaders(train_dataset, test_dataset, validation_dataset):
    """
    Create data loaders for training, testing, and validation datasets.
    :param train_dataset: Training dataset.
    :param test_dataset: Testing dataset.
    :param validation_dataset: Validation dataset.
    :return: Data loaders for training, testing, and validation.
    """
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=32, shuffle=False)
    return train_loader, test_loader, validation_loader

def generate_datasets(files):
    """
    Generate datasets from file paths.
    :param files: List of file paths for train, test, and validation data.
    :return: Datasets for training, testing, and validation.
    """
    train_dataset = Dataset(files[0])
    test_dataset = Dataset(files[1])
    validation_dataset = Dataset(files[2])
    return train_dataset, test_dataset, validation_dataset

def run_train(train_loader,
              optimizer,
              model,
              criterion,
              train_loss_values,
              train_correct_predictions,
              is_mult,
              lr_scheduler,
              train_dataset,
              lr_train_loss_values,
              lr_train_accuracies):
    """
    Run training for one epoch.
    :param train_loader: Data loader for training data.
    :param optimizer: Optimizer for updating model parameters.
    :param model: Model being trained.
    :param criterion: Loss function.
    :param train_loss_values: List to store loss values for each batch.
    :param train_correct_predictions: Counter for correct predictions.
    :param is_mult: Boolean indicating if it's multi-class classification.
    :param lr_scheduler: Learning rate scheduler (optional).
    :param train_dataset: Training dataset.
    :param lr_train_loss_values: List to store average loss for the epoch.
    :param lr_train_accuracies: List to store average accuracy for the epoch.
    """
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()

        train_loss_values.append(loss.item())
        train_correct_predictions += torch.sum(torch.argmax(outputs, dim=1) == labels).item()
    if is_mult:
        lr_scheduler.step()
    train_mean_loss = np.mean(train_loss_values)
    train_accuracy = train_correct_predictions / len(train_dataset)
    lr_train_loss_values.append(train_mean_loss)
    lr_train_accuracies.append(train_accuracy)


def run_validation(validation_loader,
                   model,
                   criterion,
                   validation_loss_values,
                   validation_correct_predictions,
                   validation_dataset,
                   lr_validation_loss_values,
                   lr_validation_accuracies):
    """
    Run validation for one epoch.
    :param validation_loader: Data loader for validation data.
    :param model: Model being validated.
    :param criterion: Loss function.
    :param validation_loss_values: List to store loss values for each batch.
    :param validation_correct_predictions: Counter for correct predictions.
    :param validation_dataset: Validation dataset.
    :param lr_validation_loss_values: List to store average loss for the epoch.
    :param lr_validation_accuracies: List to store average accuracy for the epoch.
    """
    with torch.no_grad():  # Disable gradient calculations
        for inputs, labels in validation_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            validation_loss_values.append(loss.item())
            validation_correct_predictions += torch.sum(torch.argmax(outputs, dim=1) == labels).item()

    validation_mean_loss = np.mean(validation_loss_values)
    validation_accuracy = validation_correct_predictions / len(validation_dataset)
    lr_validation_loss_values.append(validation_mean_loss)
    lr_validation_accuracies.append(validation_accuracy)

def run_test(test_loader,
             model,
             criterion,
             test_loss_values,
             test_correct_predictions,
             test_dataset,
             lr_test_loss_values,
             lr_test_accuracies):
    """
    Run testing for one epoch.
    :param test_loader: Data loader for test data.
    :param model: Model being tested.
    :param criterion: Loss function.
    :param test_loss_values: List to store loss values for each batch.
    :param test_correct_predictions: Counter for correct predictions.
    :param test_dataset: Testing dataset.
    :param lr_test_loss_values: List to store average loss for the epoch.
    :param lr_test_accuracies: List to store average accuracy for the epoch.
    """
    with torch.no_grad():  # Disable gradient calculations
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            test_loss_values.append(loss.item())
            test_correct_predictions += torch.sum(torch.argmax(outputs, dim=1) == labels).item()

    test_mean_loss = np.mean(test_loss_values)
    test_accuracy = test_correct_predictions / len(test_dataset)
    lr_test_loss_values.append(test_mean_loss)
    lr_test_accuracies.append(test_accuracy)

def print_best_validation_test_accuracy(learning_rates, end_validations_accuracies, end_test_accuracies):
    """
    Print the best validation and test accuracies for all learning rates.
    :param learning_rates: List of learning rates.
    :param end_validations_accuracies: List of validation accuracies for the last epoch.
    :param end_test_accuracies: List of test accuracies for the last epoch.
    """
    print("\n------------------Best validation and test accuracies--------------------")
    for index, lr in enumerate(learning_rates):
        print(f"Validation accuracy in last epoch with lr = {lr}: {end_validations_accuracies[index]:.6f}")
        print(f"Test accuracy in last epoch with lr = {lr}: {end_test_accuracies[index]:.6f}")
    print()

def print_best_learning_rate_and_test_by_validation(end_validations_accuracies, end_test_accuracies, learning_rates):
    """
    Print and return the best learning rate and corresponding test accuracy based on validation performance.
    :param end_validations_accuracies: List of validation accuracies for the last epoch.
    :param end_test_accuracies: List of test accuracies for the last epoch.
    :param learning_rates: List of learning rates.
    :return: Best learning rate.
    """
    max_validation_accuracy_lr = np.argmax(end_validations_accuracies)
    best_test_accuracy = end_test_accuracies[max_validation_accuracy_lr]
    best_learning_rate = learning_rates[max_validation_accuracy_lr]
    print(f"Best Validation Accuracy: {end_validations_accuracies[max_validation_accuracy_lr]:.6f}")
    print(f'Best Test Accuracy According to validation: {best_test_accuracy:.6f}')
    print(f'Best Learning Rate According to test: {best_learning_rate}')
    return best_learning_rate

def plot_decision_boundaries_with_best_learning_rate(n_classes, best_learning_rate, train_loader, files):
    """
    Plot decision boundaries for the model trained with the best learning rate.
    :param n_classes: Number of output classes.
    :param best_learning_rate: Learning rate that gave the best validation accuracy.
    :param train_loader: Data loader for training data.
    :param files: List of file paths for train, test, and validation datasets.
    """
    best_model = Logistic_Regression(2, n_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(best_model.parameters(), lr=best_learning_rate)
    best_model.train()  # set the model to training mode
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = best_model(inputs)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()
    X, y = generate_X_y_from_csv(files[2])

    helpers.plot_decision_boundaries(best_model, X, y,
                                     title=f'Decision Boundaries for model with lr ={best_learning_rate}')

def plot_multi_best_test_validation(learning_rates, end_validations_accuracies, end_test_accuracies):
    """
    Plot the best test and validation accuracies for multiple learning rates.
    :param learning_rates: List of learning rates.
    :param end_validations_accuracies: List of validation accuracies for the last epoch.
    :param end_test_accuracies: List of test accuracies for the last epoch.
    """
    plt.plot(learning_rates, end_validations_accuracies, label='Validation', marker='o')
    plt.plot(learning_rates, end_test_accuracies, label='Test', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy Value')
    plt.title(f'Validation and test accuracies during last epoch')
    plt.legend()
    plt.show()

def prepare_model(n_classes, is_mult, lr):
    """
    Prepare the model, criterion, optimizer, and (optionally) the learning rate scheduler.
    :param n_classes: Number of output classes.
    :param is_mult: Boolean indicating if it is multi-class classification.
    :param lr: Learning rate.
    :return: Tuple (model, criterion, optimizer, lr_scheduler).
    """
    model = Logistic_Regression(2, n_classes)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.3) if is_mult else None
    return model, criterion, optimizer, lr_scheduler

def plot_loss(lr_train_loss_values, lr_validation_loss_values, lr_test_loss_values, lr):
    """
    Plot loss progression for train, validation, and test datasets.
    :param lr_train_loss_values: Loss values for training.
    :param lr_validation_loss_values: Loss values for validation.
    :param lr_test_loss_values: Loss values for testing.
    :param lr: Learning rate used.
    """
    plt.plot(lr_train_loss_values, label='Train', marker='o')
    plt.plot(lr_validation_loss_values, label='Validation', marker='o')
    plt.plot(lr_test_loss_values, label='Test', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Loss Value')
    plt.title(f'Loss Progression for Learning Rate {lr}')
    plt.legend()
    plt.show()

def plot_accuracy(lr_train_accuracies, lr_validation_accuracies, lr_test_accuracies, lr):
    """
    Plot accuracy progression for train, validation, and test datasets.
    :param lr_train_accuracies: Accuracy values for training.
    :param lr_validation_accuracies: Accuracy values for validation.
    :param lr_test_accuracies: Accuracy values for testing.
    :param lr: Learning rate used.
    """
    plt.plot(lr_train_accuracies[1:], label='Train', marker='o')
    plt.plot(lr_validation_accuracies[1:], label='Validation', marker='o')
    plt.plot(lr_test_accuracies[1:], label='Test', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy Value')
    plt.title(f'Accuracy Progression for Learning Rate {lr}')
    plt.legend()
    plt.show()


def run_logistic_regression(num_epochs, files, learning_rates, is_mult=False):
    """
    Run logistic regression for a specified number of epochs and learning rates.
    :param num_epochs: Number of training epochs.
    :param files: List of file paths for train, test, and validation datasets.
    :param learning_rates: List of learning rates.
    :param is_mult: Boolean indicating if it is multi-class classification.
    """

    # Generate datasets and data loaders from the provided file paths
    train_dataset, test_dataset, validation_dataset = generate_datasets(files)
    train_loader, test_loader, validation_loader = generate_loaders(train_dataset, test_dataset, validation_dataset)

    # Determine the number of output classes (e.g., binary or multi-class)
    n_classes = len(torch.unique(train_dataset.labels))

    # Initialize lists to store accuracies for validation and test datasets across all learning rates
    end_validations_accuracies, end_test_accuracies = [], []

    # Iterate over each learning rate in the given list
    for lr in learning_rates:

        # Prepare the model, loss function (criterion), optimizer, and optional learning rate scheduler
        model, criterion, optimizer, lr_scheduler = prepare_model(n_classes, is_mult, lr)

        # Lists to track loss and accuracy for training, validation, and testing for this learning rate
        lr_train_loss_values, lr_validation_loss_values, lr_test_loss_values = [], [], []
        lr_train_accuracies, lr_validation_accuracies, lr_test_accuracies = [], [], []

        # Train the model for the specified number of epochs
        for epoch in range(num_epochs):

            # Initialize lists to store loss and correct predictions for this epoch
            train_loss_values, validation_loss_values, test_loss_values = [], [], []
            train_correct_predictions, validation_correct_predictions, test_correct_predictions = 0., 0., 0.

            # Set the model to training mode and run one training epoch
            model.train()
            run_train(train_loader, optimizer, model, criterion, train_loss_values, train_correct_predictions,
                      is_mult, lr_scheduler, train_dataset, lr_train_loss_values, lr_train_accuracies)

            # Set the model to evaluation mode for validation and test
            model.eval()
            run_validation(validation_loader, model, criterion, validation_loss_values, validation_correct_predictions,
                               validation_dataset, lr_validation_loss_values,lr_validation_accuracies)
            run_test(test_loader, model, criterion, test_loss_values, test_correct_predictions, test_dataset,
                     lr_test_loss_values, lr_test_accuracies)

        # After completing all epochs, plot the loss and accuracy progression for this learning rate
        plot_loss(lr_train_loss_values, lr_validation_loss_values, lr_test_loss_values, lr)

        plot_accuracy(lr_train_accuracies, lr_validation_accuracies, lr_test_accuracies, lr)

        # Store the final validation and test accuracies for this learning rate
        end_validations_accuracies.append(lr_validation_accuracies[-1])
        end_test_accuracies.append(lr_test_accuracies[-1])

    # Print the best validation and test accuracies across all learning rates
    print_best_validation_test_accuracy(learning_rates, end_validations_accuracies, end_test_accuracies)

    # Identify the best learning rate based on validation accuracy and its corresponding test accuracy
    best_learning_rate = (print_best_learning_rate_and_test_by_validation
                          (end_validations_accuracies, end_test_accuracies, learning_rates))

    # Plot the decision boundaries for the best learning rate
    plot_decision_boundaries_with_best_learning_rate(n_classes, best_learning_rate, train_loader, files)

    # If this is multi-class classification, also plot the best validation and test accuracies
    if is_mult:
        plot_multi_best_test_validation(learning_rates, end_validations_accuracies, end_test_accuracies)


def decision_tree(max_depth, file_name_train, file_name_test):
    """
    Train and evaluate a decision tree classifier with a specified maximum depth.
    :param max_depth: Maximum depth of the decision tree.
    :param file_name_train: File path for the training dataset.
    :param file_name_test: File path for the testing dataset.
    """

    # Load training and testing data
    train_data, _ = helpers.read_data_demo(file_name_train)
    X_train, y_train = train_data[:, :-1], train_data[:, -1]

    test_data, _ = helpers.read_data_demo(file_name_test)
    X_test, y_test = test_data[:, :-1], test_data[:, -1]

    # Initialize and train the decision tree classifier
    tree = DecisionTreeClassifier(max_depth=max_depth)
    tree.fit(X_train, y_train)

    # Predict and evaluate accuracy on the test data
    predictions = tree.predict(X_test)
    accuracy = np.mean(predictions == y_test)
    print(f'{max_depth} depth Decision Tree Accuracy: {accuracy}')

    # Visualize the decision boundaries
    helpers.plot_decision_boundaries(tree, X_test, y_test,
                                     title=f"Decision boundaries for a Decision Tree with max_depth = {max_depth} ")

def run_train_with_ridge_regularization(train_loader, len_train_dataset, optimizer, model, criterion, device_name):
    """
    Train a logistic regression model with ridge regularization.
    :param train_loader: Data loader for training data.
    :param len_train_dataset: Size of the training dataset.
    :param optimizer: Optimizer for updating model parameters.
    :param model: Model being trained.
    :param criterion: Loss function.
    :param device_name: Device to use (e.g., 'cpu' or 'cuda').
    :return: Training accuracy for the epoch.
    """
    model.train()
    correct_predictions = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device_name), labels.to(device_name)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()

        correct_predictions += (torch.argmax(outputs, dim=1) == labels).sum().item()

    accuracy = correct_predictions / len_train_dataset
    return accuracy

def run_validation_with_ridge_regularization(validation_loader, len_validation_dataset, model, device_name):
    """
    Validate a logistic regression model with ridge regularization.
    :param validation_loader: Data loader for validation data.
    :param len_validation_dataset: Size of the validation dataset.
    :param model: Model being validated.
    :param device_name: Device to use (e.g., 'cpu' or 'cuda').
    :return: Validation accuracy for the epoch.
    """
    model.eval()
    correct_predictions = 0

    with torch.no_grad():
        for inputs, labels in validation_loader:
            inputs, labels = inputs.to(device_name), labels.to(device_name)
            outputs = model(inputs)
            correct_predictions += (torch.argmax(outputs, dim=1) == labels).sum().item()

    accuracy = correct_predictions / len_validation_dataset
    return accuracy

def run_test_with_ridge_regularization(test_loader, len_test_dataset, model, device_name):
    """
    Test a logistic regression model with ridge regularization.
    :param test_loader: Data loader for testing data.
    :param len_test_dataset: Size of the testing dataset.
    :param model: Model being tested.
    :param device_name: Device to use (e.g., 'cpu' or 'cuda').
    :return: Test accuracy for the epoch.
    """
    model.eval()
    correct_predictions = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device_name), labels.to(device_name)
            outputs = model(inputs)
            correct_predictions += (torch.argmax(outputs, dim=1) == labels).sum().item()

    accuracy = correct_predictions / len_test_dataset
    return accuracy

def run_logistic_regression_with_ridge(num_epochs, files, lambdas, device_name):
    """
    Run logistic regression with ridge regularization for different lambda values.
    :param num_epochs: Number of training epochs.
    :param files: List of file paths for train, test, and validation datasets.
    :param lambdas: List of lambda values (regularization strengths).
    :param device_name: Device to use (e.g., 'cpu' or 'cuda').
    """
    train_dataset, validation_dataset, test_dataset = generate_datasets(files)
    train_loader, validation_loader, test_loader = generate_loaders(train_dataset, validation_dataset, test_dataset)

    best_lambda = None
    best_validation_accuracy = 0
    best_model = None
    n_classes = len(torch.unique(train_dataset.labels))


    for lamda in lambdas:
        print(f"-----------------lambda = {lamda}---------------------")
        model = Logistic_Regression(input_dim=2, output_dim=n_classes)
        model.to(device_name)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=0.01,
            weight_decay=lamda
        )
        validation_accuracy = None
        for epoch in range(num_epochs):
            train_accuracy = run_train_with_ridge_regularization(train_loader, len(train_dataset), optimizer, model,
                                                                 criterion, device)
            validation_accuracy = run_validation_with_ridge_regularization(validation_loader, len(validation_dataset),
                                                                           model, device)
            test_accuracy = run_test_with_ridge_regularization(test_loader,len(test_dataset), model, device)
            print(f"Train Accuracy = {train_accuracy:.6f},"
                  f"Validation Accuracy = {validation_accuracy:.6f}"
                  f", Test Accuracy = {test_accuracy:.6f}") if epoch == num_epochs - 1 else None
        if validation_accuracy > best_validation_accuracy:
            best_lambda = lamda
            best_validation_accuracy = validation_accuracy
            best_model = model

    print(f"\nBest Lambda: {best_lambda}, Best Validation Accuracy: {best_validation_accuracy:.6f}\n")


    # Plot decision boundaries for the best model
    X, y = generate_X_y_from_csv(files[2])
    helpers.plot_decision_boundaries(best_model, X, y, f"Decision Boundaries with Best Lambda = {best_lambda}")



if __name__ == '__main__':
    run_ridge_regression()
    print("\n")
    gradient_descent()
    print("\n")
    run_logistic_regression(10, BINARY_DATA, BINARY_LOGISTIC_LEARNING_RATES)
    print("\n")
    run_logistic_regression(30, MULTI_DATA, MULTI_LOGISTIC_LEARNING_RATES, True)
    print("\n")
    decision_tree(2, TRAIN_MULTI, TEST_MULTI)
    print("\n")
    decision_tree(10, TRAIN_MULTI, TEST_MULTI)
    print("\n")
    run_logistic_regression_with_ridge(
        num_epochs=10,
        files=MULTI_DATA,
        lambdas=RIDGE_LEARNING_RATES,
        device_name=device)

