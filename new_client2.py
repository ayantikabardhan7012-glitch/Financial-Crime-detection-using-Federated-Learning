import flwr as fl
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd

#pytorch model
class Net(nn.Module):
    def __init__(self, input_size=7, num_classes=2):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

#dataset cleaning
class CustomDataset(Dataset):
    def __init__(self, file_path):
        #Read 
        df = pd.read_csv(file_path)

        #Dropping irrelevant columns(TransactionID and CustomerID)
        df = df.drop(columns=["TransactionID", "CustomerID"])

        #Converting Timestamp to numeric (UNIX time)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors='coerce')
        df["Timestamp"] = df["Timestamp"].astype("int64") // 10**9

        #Encoding categorical columns
        categorical_cols = df.select_dtypes(include=["object"]).columns
        for col in categorical_cols:
            df[col] = df[col].astype("category").cat.codes

        #Spliting dataset
        self.X = torch.tensor(df.drop(columns=["FraudLabel"]).values, dtype=torch.float32)
        self.y = torch.tensor(df["FraudLabel"].values, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


#Flower Client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, trainloader, valloader):
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)

    def get_parameters(self, config=None):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        for X, y in self.trainloader:
            self.optimizer.zero_grad()
            outputs = self.model(X)
            loss = self.criterion(outputs, y)
            loss.backward()
            self.optimizer.step()
        return self.get_parameters(), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        correct, total = 0, 0
        loss_total = 0.0
        with torch.no_grad():
            for X, y in self.valloader:
                outputs = self.model(X)
                loss = self.criterion(outputs, y)
                loss_total += loss.item() * X.size(0)
                _, predicted = torch.max(outputs, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        accuracy = correct / total
        loss = loss_total / total
        return float(loss), len(self.valloader.dataset), {"accuracy": accuracy}


if __name__ == "__main__":
    trainloader = DataLoader(CustomDataset("datasets\client2_dataset.csv"), batch_size=32, shuffle=True)
    valloader = DataLoader(CustomDataset("datasets\client2_dataset.csv"), batch_size=32)

    model = Net()

    client = FlowerClient(model, trainloader, valloader)

    fl.client.start_client(
        server_address="localhost:8080",
        client=client.to_client()
    )
