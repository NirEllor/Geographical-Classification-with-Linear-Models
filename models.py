
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
import helpers

TRAIN_CSV = 'train.csv'
VALIDATION_CSV = 'validation.csv'
TEST_CSV = 'test.csv'

TRAIN_MULTI = 'train_multiclass.csv'
VALIDATION_MULTI = 'validation_multiclass.csv'
TEST_MULTI = 'test_multiclass.csv'

RIDGE_LEARNING_RATES = [0, 2, 4, 8, 10]
BINARY_LOGISTIC_LEARNING_RATES = [0.1, 0.01, 0.001]
BINARY_DATA = [TRAIN_CSV, VALIDATION_CSV, TEST_CSV]
MULTI_LOGISTIC_LEARNING_RATES = [0.01, 0.001, 0.0003]
MULTI_DATA = [TRAIN_MULTI, VALIDATION_MULTI,  TEST_MULTI]


torch.manual_seed(42)
np.random.seed(42)
torch.set_printoptions(5)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    data, _ = helpers.read_data_demo(file)

    # Extract features and labels
    X, y = data[:, :-1], data[:, -1]
    return X, y

def process_file(file, file_index, X_train, y_train, accuracies):
    file_name = file.split(".", 1)[0].capitalize()
    # print(f"---------------- {file_name} -----------------")
    X, y = generate_X_y_from_csv(file)

    for choice_index, choice in enumerate(RIDGE_LEARNING_RATES):
        accuracy = train_and_evaluate_ridge(choice, X_train, y_train, X, y)
        accuracies[file_index, choice_index] = accuracy
        # print(f"Accuracy for lambda = {choice}: {accuracy}")

def train_and_evaluate_ridge(lambda_value, X_train, y_train, X, y):
    ridge_regression = Ridge_Regression(lambda_value)
    ridge_regression.fit(X_train, y_train)
    y_pred = ridge_regression.predict(X)

    # Calculate accuracy
    pure_accuracy = np.mean(y_pred == y)
    return round(pure_accuracy, 6)

def analyze_validation_results(validation_sublist):
    max_index = np.argmax(validation_sublist)
    min_index = np.argmin(validation_sublist)
    return max_index, min_index

def train_and_plot(max_lambda, min_lambda):
    ridge_regression_max = Ridge_Regression(max_lambda)
    ridge_regression_min = Ridge_Regression(min_lambda)

    X, y = generate_X_y_from_csv(TEST_CSV)

    ridge_regression_max.fit(X, y)
    ridge_regression_min.fit(X, y)

    helpers.plot_decision_boundaries(ridge_regression_max, X, y,
                             f"Decision Boundaries Using best lambda = {max_lambda}")
    helpers.plot_decision_boundaries(ridge_regression_min, X, y,
                             f"Decision Boundaries Using worst lambda = {min_lambda}")

def plot_accuracies(accuracies):
    plt.figure(figsize=(10, 6))
    for file_index, file in enumerate(BINARY_DATA):
        file_name = file.split(".", 1)[0].capitalize()
        plt.plot(RIDGE_LEARNING_RATES, accuracies[file_index], label=file_name, marker='o')
    # Customize the plot
    plt.xlabel("Lambda")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Lambda for Different Datasets")

    # Set custom x-axis ticks
    plt.xticks(RIDGE_LEARNING_RATES, [f"{choice}" for choice in RIDGE_LEARNING_RATES])  # Format as 0.0, 0.2, etc.

    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()


def gradient_descent():
    # Define the function and its gradient
    def gradient_f(X, Y):
        df_dx = 2 * (X - 3)
        df_dy = 2 * (Y - 5)
        return np.array([df_dx, df_dy])

    # Parameters
    learning_rate = 0.1
    iterations = 1000

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

    # Plot the optimization trajectory
    plt.figure(figsize=(8, 6))
    plt.scatter(x_vals, y_vals, c=range(iterations + 1), cmap='viridis', s=10)
    plt.colorbar(label='Iteration')
    plt.title("Gradient Descent Trajectory")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid()
    # Annotate the final point
    final_point = (float(round(x_vals[-1], 2)), float(round(y_vals[-1], 2)))
    plt.text(final_point[0] - 1, final_point[1], f"Optimal Solution: {final_point}", fontsize=15, color='blue', ha='center')

    plt.show()

    # Print the final point
    final_point = (x_vals[-1], y_vals[-1])
    print(f"Final point is {final_point[0]:.6f}, {final_point[1]:.6f}")




class Logistic_Regression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Logistic_Regression, self).__init__()

        ########## YOUR CODE HERE ##########

        # define a linear operation.
        self.linear = nn.Linear(in_features=input_dim, out_features=output_dim)
        # One output for binary classification

        ####################################


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
        """
        x = torch.from_numpy(x).float().to(self.linear.weight.data.device)
        x = self.forward(x)
        x = nn.functional.softmax(x, dim=1)
        x = x.detach().cpu().numpy()
        x = np.argmax(x, axis=1)
        return x


class Dataset(torch.utils.data.Dataset):
    """
    Any dataset should inherit from torch.utils.data.Dataset and override the __len__ and __getitem__ methods.
    __init__ is optional.
    __len__ should return the size of the dataset.
    __getitem__ should return a tuple (data, label) for the given index.
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
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def generate_loaders(train_dataset, test_dataset, validation_dataset):
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=32, shuffle=False)
    return train_loader, test_loader, validation_loader

def generate_datasets(files):
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

    # print(f'Train Accuracy: {train_accuracy}, Train Loss: {train_mean_loss.item():.6f}')

def run_validation(validation_loader,
                   model,
                   criterion,
                   validation_loss_values,
                   validation_correct_predictions,
                   validation_dataset,
                   lr_validation_loss_values,
                   lr_validation_accuracies):
    with torch.no_grad():  # Disable gradient calculations
        for inputs, labels in validation_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            # outputs = nn.functional.softmax(outputs, dim=1)
            loss = criterion(outputs.squeeze(), labels)
            validation_loss_values.append(loss.item())
            validation_correct_predictions += torch.sum(torch.argmax(outputs, dim=1) == labels).item()

    validation_mean_loss = np.mean(validation_loss_values)
    validation_accuracy = validation_correct_predictions / len(validation_dataset)
    lr_validation_loss_values.append(validation_mean_loss)
    lr_validation_accuracies.append(validation_accuracy)
    # print(f'Validation Accuracy: {validation_accuracy}, Validation Loss: {validation_mean_loss:.6f}')

def run_test(test_loader,
             model,
             criterion,
             test_loss_values,
             test_correct_predictions,
             test_dataset,
             lr_test_loss_values,
             lr_test_accuracies):
    with torch.no_grad():  # Disable gradient calculations
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            # outputs = nn.functional.softmax(outputs, dim=1)
            loss = criterion(outputs.squeeze(), labels)
            test_loss_values.append(loss.item())
            test_correct_predictions += torch.sum(torch.argmax(outputs, dim=1) == labels).item()

    test_mean_loss = np.mean(test_loss_values)
    test_accuracy = test_correct_predictions / len(test_dataset)
    lr_test_loss_values.append(test_mean_loss)
    lr_test_accuracies.append(test_accuracy)
    # print(f'Test Accuracy: {test_accuracy}, Test Loss: {test_mean_loss:.6f}')

def print_best_validation_test_accuracy(learning_rates, end_validations_accuracies, end_test_accuracies):
    print("\n------------------Best validation and test accuracies--------------------")
    for index, lr in enumerate(learning_rates):
        print(f"Validation accuracy in last epoch with lr = {lr}: {end_validations_accuracies[index]:.6f}")
        print(f"Test accuracy in last epoch with lr = {lr}: {end_test_accuracies[index]:.6f}")
    print()

def print_best_learning_rate_and_test_by_validation(end_validations_accuracies, end_test_accuracies, learning_rates):
    max_validation_accuracy_lr = np.argmax(end_validations_accuracies)
    best_test_accuracy = end_test_accuracies[max_validation_accuracy_lr]
    best_learning_rate = learning_rates[max_validation_accuracy_lr]
    print(f"Best Validation Accuracy: {end_validations_accuracies[max_validation_accuracy_lr]:.6f}")
    print(f'Best Test Accuracy According to validation: {best_test_accuracy:.6f}')
    print(f'Best Learning Rate According to test: {best_learning_rate}')
    return best_learning_rate

def plot_decision_boundaries_with_best_learning_rate(n_classes, best_learning_rate, train_loader, files):
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

    X, y = generate_X_y_from_csv(files[0])

    helpers.plot_decision_boundaries(best_model, X, y,
                                     title=f'Decision Boundaries for model with lr ={best_learning_rate}')

def plot_multi_best_test_validation(learning_rates, end_validations_accuracies, end_test_accuracies):
    plt.plot(learning_rates, end_validations_accuracies, label='Validation', marker='o')
    plt.plot(learning_rates, end_test_accuracies, label='Test', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy Value')
    plt.title(f'Validation and test accuracies during last epoch')
    plt.legend()
    plt.show()

def prepare_model(n_classes, is_mult, lr):
    # print(f"\n------------------------ learning rate is {lr} ---------------------------- ")
    model = Logistic_Regression(2, n_classes)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.3) if is_mult else None
    return model, criterion, optimizer, lr_scheduler

def run_logistic_regression(num_epochs, files, learning_rates, is_mult=False):
    train_dataset, test_dataset, validation_dataset = generate_datasets(files)
    n_classes = len(torch.unique(train_dataset.labels))
    train_loader, test_loader, validation_loader = generate_loaders(train_dataset, test_dataset, validation_dataset)
    end_validations_accuracies, end_test_accuracies = [], []
    for lr in learning_rates:
        model, criterion, optimizer, lr_scheduler = prepare_model(n_classes, is_mult, lr)
        # Train the model for a few epochs with GPU acceleration
        lr_train_loss_values, lr_validation_loss_values, lr_test_loss_values = [], [], []
        lr_train_accuracies, lr_validation_accuracies, lr_test_accuracies = [], [], []
        for epoch in range(num_epochs + 1):
            train_loss_values, validation_loss_values, test_loss_values = [], [], []
            train_correct_predictions, validation_correct_predictions, test_correct_predictions = 0., 0., 0.
            epoch_number = f" Epoch {epoch}" if epoch else "initialization"
            # print(f"\n------------------------ {epoch_number } ---------------------------- ")
            model.train()  # set the model to training mode
            run_train(train_loader, optimizer, model, criterion, train_loss_values, train_correct_predictions,
                      is_mult, lr_scheduler, train_dataset, lr_train_loss_values, lr_train_accuracies)
            model.eval()  # Set the model to evaluation mode
            run_validation(validation_loader, model, criterion, validation_loss_values, validation_correct_predictions,
                               validation_dataset, lr_validation_loss_values,lr_validation_accuracies)
            run_test(test_loader, model, criterion, test_loss_values, test_correct_predictions, test_dataset,
                     lr_test_loss_values, lr_test_accuracies)
        # Plot the loss values through epochs
        plot_loss(lr_train_loss_values, lr_validation_loss_values, lr_test_loss_values, lr)
        # Plot the accuracy values through epochs
        plot_accuracy(lr_train_accuracies, lr_validation_accuracies, lr_test_accuracies, lr)
        end_validations_accuracies.append(lr_validation_accuracies[-2])
        end_test_accuracies.append(lr_test_accuracies[-2])
    print_best_validation_test_accuracy(learning_rates, end_validations_accuracies, end_test_accuracies)
    best_learning_rate = (print_best_learning_rate_and_test_by_validation
                          (end_validations_accuracies, end_test_accuracies, learning_rates))
    plot_decision_boundaries_with_best_learning_rate(n_classes, best_learning_rate, train_loader, files)
    if is_mult:
        plot_multi_best_test_validation(learning_rates, end_validations_accuracies, end_test_accuracies)


def plot_loss(lr_train_loss_values, lr_validation_loss_values, lr_test_loss_values, lr):
    plt.plot(lr_train_loss_values[1:], label='Train', marker='o')
    plt.plot(lr_validation_loss_values[1:], label='Validation', marker='o')
    plt.plot(lr_test_loss_values[1:], label='Test', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Loss Value')
    plt.title(f'Loss Progression for Learning Rate {lr}')
    plt.legend()
    plt.show()

def plot_accuracy(lr_train_accuracies, lr_validation_accuracies, lr_test_accuracies, lr):
    plt.plot(lr_train_accuracies[1:], label='Train', marker='o')
    plt.plot(lr_validation_accuracies[1:], label='Validation', marker='o')
    plt.plot(lr_test_accuracies[1:], label='Test', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy Value')
    plt.title(f'Accuracy Progression for Learning Rate {lr}')
    plt.legend()
    plt.show()


def decision_tree(max_depth, file_name_train, file_name_test):
    train_data, _ = helpers.read_data_demo(file_name_train)
    X_train, y_train = train_data[:, :-1], train_data[:, -1]
    test_data, _ = helpers.read_data_demo(file_name_test)
    X_test, y_test = test_data[:, :-1], test_data[:, -1]
    tree = helpers.DecisionTreeClassifier(max_depth=max_depth)
    tree.fit(X_train, y_train)
    predictions = tree.predict(X_test)
    accuracy = np.mean(predictions == y_test)
    print(f'{max_depth} depth Decision Tree Accuracy: {accuracy}')

    helpers.plot_decision_boundaries(tree, X_train, y_train,
                                     title=f"Decision boundaries for a Decision Tree with max_depth = {max_depth} ")

def run_train_with_ridge_regularization(train_loader, len_train_dataset, optimizer, model, criterion, device):
    """
    Train the logistic regression model with ridge regularization.
    """
    model.train()
    correct_predictions = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()

        correct_predictions += (torch.argmax(outputs, dim=1) == labels).sum().item()

    accuracy = correct_predictions / len_train_dataset
    return accuracy

def run_validation_with_ridge_regularization(validation_loader, len_validation_dataset, model, device):
    """
    Validate the logistic regression model with ridge regularization.
    """
    model.eval()
    correct_predictions = 0

    with torch.no_grad():
        for inputs, labels in validation_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            correct_predictions += (torch.argmax(outputs, dim=1) == labels).sum().item()

    accuracy = correct_predictions / len_validation_dataset
    return accuracy

def run_test_with_ridge_regularization(test_loader, len_test_dataset, model, device):
    """
    Test the logistic regression model with ridge regularization.
    """
    model.eval()
    correct_predictions = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            correct_predictions += (torch.argmax(outputs, dim=1) == labels).sum().item()

    accuracy = correct_predictions / len_test_dataset
    return accuracy

def run_logistic_regression_with_ridge(num_epochs, files, lambdas, device):
    """
    Run logistic regression with ridge regularization.
    """
    train_dataset, validation_dataset, test_dataset = generate_datasets(files)
    train_loader, validation_loader, test_loader = generate_loaders(train_dataset, validation_dataset, test_dataset)

    best_lambda = None
    best_validation_accuracy = 0
    best_model = None
    criterion = None
    n_classes = len(torch.unique(train_dataset.labels))


    for lamda in lambdas:
        print(f"-----------------lambda = {lamda}---------------------")
        model = Logistic_Regression(input_dim=2, output_dim=n_classes)
        model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=0.01,
            weight_decay=lamda
        )
        validation_accuracy = None
        for epoch in range(num_epochs):
            train_accuracy = run_train_with_ridge_regularization(train_loader, len(train_dataset), optimizer, model, criterion, device)
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
    X, y = generate_X_y_from_csv(files[0])
    helpers.plot_decision_boundaries(best_model, X, y, f"Decision Boundaries with Best Lambda = {best_lambda}")



if __name__ == '__main__':
    run_ridge_regression()
    # print("\n")
    # gradient_descent()
    # print("\n")
    # run_logistic_regression(10, BINARY_DATA, BINARY_LOGISTIC_LEARNING_RATES)
    # print("\n")
    # run_logistic_regression(30, MULTI_DATA, MULTI_LOGISTIC_LEARNING_RATES, True)
    # print("\n")
    decision_tree(2, TRAIN_MULTI, TEST_MULTI)
    # print("\n")
    # decision_tree(10, TRAIN_MULTI, TEST_MULTI)
    # print("\n")
    # run_logistic_regression_with_ridge(
    #     num_epochs=10,
    #     files=MULTI_DATA,
    #     lambdas=RIDGE_LEARNING_RATES,
    #     device=device)

