import torch
from torchmetrics.classification import BinaryAccuracy

class Linear(torch.nn.Module):
    def __init__(self, n_x, n_y):
        super().__init__()
        self.fc = torch.nn.Linear(n_x, n_y, dtype=torch.float64)

    def forward(self, x):
        return self.fc(x)

class ReLU(torch.nn.Module):
    def __init__(self, n_x, n_y):
        super().__init__()
        self.fc = torch.nn.Linear(n_x, n_y, dtype=torch.float64)

    def forward(self, x):
        return torch.nn.functional.relu(self.fc(x))

class Sigmoid(torch.nn.Module):
    def __init__(self, n_x, n_y):
        super().__init__()
        self.fc = torch.nn.Linear(n_x, n_y, dtype=torch.float64)

    def forward(self, x):
        return torch.sigmoid(self.fc(x))
    

class DeepNeuralNet(torch.nn.Module):
    def __init__(self, n_x, n_h, n_y):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.loss = torch.nn.BCELoss()
        self.accuracy = BinaryAccuracy()
        self.create_layers(n_x, n_h, n_y)

    def create_layers(self, n_x, n_h, n_y):
        layer_sizes = [n_x] + n_h + [n_y]
        for i in range(len(layer_sizes) - 1):
            if i == len(layer_sizes) - 2:
                self.layers.append(Sigmoid(layer_sizes[i], layer_sizes[i + 1]))
            else:
                self.layers.append(ReLU(layer_sizes[i], layer_sizes[i + 1]))
                self.layers.append(torch.nn.Dropout(0.5))
                self.layers.append(torch.nn.LeakyReLU(negative_slope=0.1))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def trainer(self, inputs, outputs, val_inputs, val_outputs, N_max = 1000, lr = 0.01, batch_size = 32):
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
        self.eval()
        with torch.no_grad():
            pred = self(inputs)
            loss = self.loss(pred, outputs)
            acc = self.accuracy(pred, outputs).item()
        print("Test Loss = {} - Test Accuracy = {}".format(loss, acc))
        return loss, acc