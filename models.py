import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np


def learn_classifier(data_loader_train, data_loader_test, device='cpu'):
    class mlp(nn.Module):
        def __init__(self, input_size):
            super(mlp, self).__init__()
            #self.conv1 = nn.Conv1d(1, 50, 4)
            #self.conv2 = nn.Conv1d(50, 100, 4)
            #self.pool = nn.MaxPool1d(4)
            self.l1 = nn.Linear(20, 128)
            self.l2 = nn.Linear(128, 64)
            self.l3 = nn.Linear(64, 32)
            self.l4 = nn.Linear(32, 1)
            self.relu = nn.ReLU()
            self.flatten = nn.Flatten()
            self.dropout = nn.Dropout(0.5)

        def forward(self, x):
            x = self.flatten(x)
            #x = self.pool(self.conv1(x))
            #x = self.conv2(x).squeeze()
            x = self.dropout(self.relu(self.l1(x)))
            x = self.dropout(self.relu(self.l2(x)))
            x = self.dropout(self.relu(self.l3(x)))
            x = self.l4(x)
            return x

    def binary_acc(y_pred, y_test):
        y_pred_tag = torch.round(torch.sigmoid(y_pred))
        correct_results_sum = (y_pred_tag == y_test).sum().float()
        acc = correct_results_sum / y_test.shape[0]
        return torch.round(acc * 100)

    def learn(alpha=0.0001):
        input_size = np.prod(next(iter(data_loader_train))[0][0].shape)

        classifier = mlp(input_size)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(classifier.parameters(), lr=0.001)

        classifier.train()
        classifier.to(device)

        epochs = 20
        for epoch in range(1, epochs):
            epoch_loss = 0
            epoch_acc = 0
            for X, y in data_loader_train:
                optimizer.zero_grad()
                X, y = X.to(device), y.to(device)
                y_hat = classifier(X)
                params = torch.cat([x.view(-1) for x in classifier.parameters()])
                l2_regularization = alpha * torch.norm(params, 2) ** 2
                loss = criterion(y_hat, y.unsqueeze(1)) + l2_regularization
                loss.backward()
                optimizer.step()

                acc = binary_acc(y_hat, y.unsqueeze(1)).to('cpu')

                epoch_loss += loss.item()
                epoch_acc += acc.item()
            if epoch % 20 == 0:
                print(f'Epoch {epoch} acc is {epoch_acc/ len(data_loader_train)}')

        classifier.eval()
        test_acc = 0
        for X, y in data_loader_test:
            X, y = X.to(device), y.to(device)
            y_hat = classifier(X)
            test_acc += binary_acc(y_hat, y.unsqueeze(1))

        print(f'Classifier learn has finished with acc: {epoch_acc / len(data_loader_train):.3f}')
        print(f'Test acc: {test_acc / len(data_loader_test):.3f}')
        return classifier, (test_acc.item() / len(data_loader_test))
    classifier_raw, acc_raw = learn(0)
    classifier, acc = learn()
    return classifier, classifier_raw, acc, acc_raw


def learn_regressor(data_loader_train, device):
    class mlp_r(nn.Module):
        def __init__(self):
            super(mlp_r, self).__init__()
            self.l1 = nn.LazyLinear(128)
            self.l2 = nn.Linear(128, 64)
            self.l3 = nn.Linear(64, 32)
            self.l4 = nn.Linear(32, 1)

            self.relu = nn.ReLU()
            self.flatten = nn.Flatten()

        def forward(self, x):
            x = self.flatten(x)
            x = self.relu(self.l1(x))
            x = self.relu(self.l2(x))
            x = self.relu(self.l3(x))
            x = self.relu(self.l4(x))
            return x

    input_size = np.prod(next(iter(data_loader_train))[0].shape)

    regressor = mlp_r()
    criterion = nn.MSELoss()
    opt = optim.Adam(regressor.parameters(), lr=0.001)

    regressor.train()
    regressor.to(device)

    epochs = 200

    for epoch in range(1, epochs):
        epoch_loss = 0
        epoch_acc = 0
        for X, y in data_loader_train:
            opt.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = regressor(X)
            loss = criterion(y_hat, y.unsqueeze(1))
            loss.backward()
            opt.step()
            epoch_loss += loss.item()

    print(f'Regressor learning has finished with loss: {epoch_loss / len(data_loader_train):.5f}')

    return regressor


def get_preds_regressor(dataloader, model, device):
    model.eval()

    y_preds = list()
    y_true = list()
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y_hat = model(X).cpu().numpy()
            y_preds.append(y_hat)

    return np.concatenate(y_preds, axis=0).squeeze()
