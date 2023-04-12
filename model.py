import torch
from torchmetrics.classification import BinaryAccuracy

# Linear layer with input size n_x and output size n_y.
class Linear(torch.nn.Module):
    def __init__(self, n_x, n_y):
        super().__init__()
        self.fc = torch.nn.Linear(n_x, n_y, dtype=torch.float64)

    def forward(self, x):
        return self.fc(x)

# Linear layer with input size n_x and output size n_y. Rectified Linear Unit (ReLU) activation function is applied.
class ReLU(torch.nn.Module):
    def __init__(self, n_x, n_y):
        super().__init__()
        self.fc = torch.nn.Linear(n_x, n_y, dtype=torch.float64)

    def forward(self, x):
        return torch.nn.functional.relu(self.fc(x))

# Linear layer with input size n_x and output size n_y. Sigmoid activation function is applied.
class Sigmoid(torch.nn.Module):
    def __init__(self, n_x, n_y):
        super().__init__()
        self.fc = torch.nn.Linear(n_x, n_y, dtype=torch.float64)

    def forward(self, x):
        return torch.sigmoid(self.fc(x))
    

class DeepNeuralNet(torch.nn.Module):
    def __init__(self, n_x, n_h, n_y):
        """
        This function initializes a neural network with a specified number of input, hidden, and output
        layers, and sets up the loss and accuracy metrics.
        
        :param n_x: The number of input features or dimensions of the input data
        :param n_h: n_h is the number of neurons in the hidden layer of a neural network. It is a
        hyperparameter that can be adjusted to optimize the performance of the network. Increasing the
        number of neurons in the hidden layer can increase the capacity of the network to learn complex
        patterns in the data, but it can
        :param n_y: The parameter n_y represents the number of output units in the neural network. It is
        the dimensionality of the output layer
        """
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.loss = torch.nn.BCELoss()
        self.accuracy = BinaryAccuracy()
        self.create_layers(n_x, n_h, n_y)

    def create_layers(self, n_x, n_h, n_y):
        """
        :param n_x: The number of input features or input neurons in the neural network
        :param n_h: n_h is a list of integers representing the number of neurons in each hidden layer of
        a neural network
        :param n_y: n_y is the number of neurons in the output layer of the neural network. It
        represents the number of classes or the number of values that the network is expected to predict
        """
        layer_sizes = [n_x] + n_h + [n_y]
        for i in range(len(layer_sizes) - 1):
            if i == len(layer_sizes) - 2:
                self.layers.append(Sigmoid(layer_sizes[i], layer_sizes[i + 1]))
            else:
                self.layers.append(ReLU(layer_sizes[i], layer_sizes[i + 1]))
                self.layers.append(torch.nn.Dropout(0.5))
                self.layers.append(torch.nn.LeakyReLU(negative_slope=0.1))

    def forward(self, x):
        """
        This function applies a forward pass through all layers in a neural network and returns the
        output.
        
        :param x: The input data to be passed through the layers of the neural network. It could be a
        single data point or a batch of data points
        :return: The output of the neural network after passing the input `x` through all the layers in
        `self.layers`.
        """
        for layer in self.layers:
            x = layer(x)
        return x
    
    def trainer(self, inputs, outputs, val_inputs, val_outputs, N_max = 1000, lr = 0.01, batch_size = 32):
        """
        This function trains a neural network model using stochastic gradient descent and returns the
        training and validation loss and accuracy history.
        
        :param inputs: The input data for the model training
        :param outputs: The expected outputs for the given inputs
        :param val_inputs: val_inputs is a tensor containing the validation set inputs. It is used to
        evaluate the performance of the model on a separate set of data during training to prevent
        overfitting
        :param val_outputs: val_outputs is a tensor containing the expected outputs for the validation
        set. It is used to calculate the validation loss and accuracy during the training process
        :param N_max: The maximum number of iterations for training the model, defaults to 1000
        (optional)
        :param lr: lr stands for learning rate, which is a hyperparameter that determines the step size
        at each iteration while moving toward a minimum of a loss function during training. It controls
        how much the model weights are updated in response to the estimated error each time the model is
        updated. A higher learning rate can result in
        :param batch_size: The number of samples in each batch of data that is used to update the model
        during training, defaults to 32.
        :return: four lists: `train_loss_history`, `validation_loss_history`, `train_accuracy_history`,
        and `validation_accuracy_history`. These lists contain the loss and accuracy values for the
        training and validation sets at each iteration of the training process.
        """
        dataset = torch.utils.data.TensorDataset(inputs, outputs)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = True)
        optimizer = torch.optim.SGD(self.parameters(), lr = lr)
        optimizer.zero_grad()
        self.train_loss_history = []
        self.train_accuracy_history = []
        self.validation_loss_history = []
        self.validation_accuracy_history = []
        for iteration_number in range(1, N_max + 1):
            for batch, (X, y) in enumerate(data_loader):
                pred = self(X)
                total_loss = self.loss(pred, y)
                total_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            prediction = self(inputs)
            acc = self.accuracy(prediction, outputs).item()
            self.train_accuracy_history.append(acc)
            self.train_loss_history.append(total_loss.item())
            with torch.no_grad():
                pred = self(val_inputs)
                val_loss = self.loss(pred, val_outputs)
                val_acc = self.accuracy(pred, val_outputs).item()
            self.validation_loss_history.append(val_loss.item())
            self.validation_accuracy_history.append(val_acc)
            print("Iteration {} - Loss = {} - Accuracy = {} - Validation Loss = {} - Validation Accuracy = {}".format(iteration_number, total_loss, acc, val_loss, val_acc))
        return self.train_loss_history, self.validation_loss_history, self.train_accuracy_history, self.validation_accuracy_history
    
    def tester(self, inputs, outputs):
        """
        This function evaluates the performance of a neural network model on a test dataset and returns
        the loss and accuracy.
        
        :param inputs: The input data to the model for testing. This could be a batch of images, text,
        or any other type of data that the model is designed to process
        :param outputs: The expected output values for the given inputs. These are the ground truth
        values that the model is trying to predict
        :return: the test loss and test accuracy of a model on a given set of inputs and outputs.
        """
        self.eval()
        with torch.no_grad():
            pred = self(inputs)
            loss = self.loss(pred, outputs)
            acc = self.accuracy(pred, outputs).item()
        print("Test Loss = {} - Test Accuracy = {}".format(loss, acc))
        return loss, acc