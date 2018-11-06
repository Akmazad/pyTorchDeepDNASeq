import numpy as np

import torch
import torch.utils.data as tdata
import torch.nn as tnn
import torch.nn.functional as tfunc
import torch.optim as topti

class toyDataset(tdata.Dataset):
    def __init__(self, dataFile, labelFile):
        self.inputs = np.loadtxt(dataFile, dtype = np.float32).reshape(-1, 4, 1000)
        self.labels = np.loadtxt(labelFile, dtype = np.float32)

        self.length = len(self.labels)

    def __getitem__(self, index):
        inputSample = self.inputs[index]
        labelSample = self.labels[index]
        sample = {"input": inputSample, "label": labelSample}

        return sample

    def __len__(self):

        return self.length

class network(tnn.Module):

    def __init__(self):
        super(network, self).__init__()

        self.conv1 = tnn.Conv1d(4, 32, 4)
        self.conv2 = tnn.Conv1d(32, 64, 4)
        self.conv3 = tnn.Conv1d(64, 128, 4)

        self.fc1 = tnn.Linear(15616, 128)
        self.fc2 = tnn.Linear(128, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = tfunc.leaky_relu(x, 0.1)
        x = tfunc.max_pool1d(x, 2)
        x = tfunc.dropout(x, 0.2)

        x = self.conv2(x)
        x = tfunc.leaky_relu(x, 0.1)
        x = tfunc.max_pool1d(x, 2)
        x = tfunc.dropout(x, 0.2)

        x = self.conv3(x)
        x = tfunc.leaky_relu(x, 0.1)
        x = tfunc.max_pool1d(x, 2)
        x = tfunc.dropout(x, 0.2)

        x = x.view(x.shape[0], -1)

        x = self.fc1(x)
        x = tfunc.leaky_relu(x, 0.1)
        x = tfunc.dropout(x, 0.2)

        x = self.fc2(x)

        x = x.view(-1)

        return x

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device: " + str(device))

    trainDataset = toyDataset("toy_TrainData.csv", "toy_TrainLabel.csv")
    trainLoader = tdata.DataLoader(dataset = trainDataset, batch_size = 16, shuffle = True)

    net = network().to(device)
    criterion = tnn.BCEWithLogitsLoss()
    optimiser = topti.Adam(net.parameters(), lr = 0.001)

    for epoch in range(5):
        runningLoss = 0

        for i, batch in enumerate(trainLoader):
            inputs, labels = batch["input"].to(device), batch["label"].to(device)

            optimiser.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimiser.step()

            runningLoss += loss.item()

            if i % 32 == 31:
                print("Epoch: %2d, Batch: %4d, Loss: %.3f" % (epoch + 1, i + 1, runningLoss / 32))
                runningLoss = 0

    testDataset = toyDataset("toy_TestData.csv", "toy_TestLabel.csv")
    testLoader = tdata.DataLoader(dataset = testDataset, batch_size = 16)

    truePos, trueNeg, falsePos, falseNeg = 0, 0, 0, 0

    with torch.no_grad():
        for batch in testLoader:
            inputs, labels = batch["input"].to(device), batch["label"].to(device)

            outputs = torch.sigmoid(net(inputs))
            predicted = torch.round(outputs)

            truePos += torch.sum(labels * predicted).item()
            trueNeg += torch.sum((1 - labels) * (1 - predicted)).item()
            falsePos += torch.sum((1 - labels) * predicted).item()
            falseNeg += torch.sum(labels * (1 - predicted)).item()

    accuracy = 100 * (truePos + trueNeg) / len(testDataset)
    matthews = MCC(truePos, trueNeg, falsePos, falseNeg)

    print("Classification accuracy: %.2f%%\n"
          "Matthew Correlation Coefficient: %.2f" % (accuracy, matthews))

def MCC(tp, tn, fp, fn):
    numerator = tp * tn - fp * fn
    denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5

    with np.errstate(divide = "ignore", invalid = "ignore"):
        return np.divide(numerator, denominator)

if __name__ == '__main__':
    main()