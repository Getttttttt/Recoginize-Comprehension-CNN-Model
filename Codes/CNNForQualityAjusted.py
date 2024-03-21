import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchviz import make_dot
from captum.attr import IntegratedGradients
from torch.autograd import Variable


csv_path = './Dataset/ENData.csv'
df = pd.read_csv(csv_path)

train_df, test_df = train_test_split(df, test_size=0.2, random_state=3407)


class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        identifier = tuple(self.dataframe.iloc[idx, 0:8].values.tolist() + [self.dataframe.iloc[idx, 26]])
        
        channel1 = self.dataframe.iloc[idx, 1:7].values.astype('float32')
        channel2 = self.dataframe.iloc[idx, 7:9].values.astype('float32')
        channel3 = self.dataframe.iloc[idx, 12:19].values.astype('float32')  
        channel4 = self.dataframe.iloc[idx, 19:22].values.astype('float32')
        channel5 = self.dataframe.iloc[idx, [26] + list(range(22, 25))].values.astype('float32')
        channel6 = self.dataframe.iloc[idx, 27:38].values.astype('float32')
        channel7 = self.dataframe.iloc[idx, 38:40].values.astype('float32')
        output_label = self.dataframe.iloc[idx, 25].astype('float32')

        if self.transform:
            channel1 = self.transform(channel1)
            channel2 = self.transform(channel2)
            channel3 = self.transform(channel3)
            channel4 = self.transform(channel4)
            channel5 = self.transform(channel5)
            channel6 = self.transform(channel6)
            channel7 = self.transform(channel7)

        channel1 = torch.from_numpy(channel1)
        channel2 = torch.from_numpy(channel2)
        channel3 = torch.from_numpy(channel3)
        channel4 = torch.from_numpy(channel4)
        channel5 = torch.from_numpy(channel5)
        channel6 = torch.from_numpy(channel6)
        channel7 = torch.from_numpy(channel7)
        output_label = torch.tensor(output_label, dtype=torch.float32)

        return identifier, (channel1, channel2, channel3, channel4, channel5, channel6, channel7), output_label


class CustomTransform:
    def __call__(self, input_features):
        input_features = (input_features - 0.5) / 0.5 
        return input_features

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1_1 = nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool1d(2, 2)

        self.fc1 = nn.Linear(448, 120)  
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        channel1, channel2, channel3, channel4, channel5, channel6, channel7 = x
        
        channel1 = channel1.float()
        channel2 = channel2.float()
        channel3 = channel3.float()
        channel4 = channel4.float()
        channel5 = channel5.float()
        channel6 = channel6.float()
        channel7 = channel7.float()

        channel1 = channel1.unsqueeze(1)  
        channel1 = F.relu(self.conv1_1(channel1))
        channel1 = F.relu(self.conv1_2(channel1))
        channel1 = self.pool(channel1)

        channel2 = channel2.unsqueeze(1)
        channel2 = F.relu(self.conv1_1(channel2))
        channel2 = F.relu(self.conv1_2(channel2))
        channel2 = self.pool(channel2)

        channel3 = channel3.unsqueeze(1)
        channel3 = F.relu(self.conv1_1(channel3))
        channel3 = F.relu(self.conv1_2(channel3))
        channel3 = self.pool(channel3)

        channel4 = channel4.unsqueeze(1)
        channel4 = F.relu(self.conv1_1(channel4))
        channel4 = F.relu(self.conv1_2(channel4))
        channel4 = self.pool(channel4)
        
        channel5 = channel5.unsqueeze(1)  
        channel5 = F.relu(self.conv1_1(channel5))
        channel5 = F.relu(self.conv1_2(channel5))
        channel5 = self.pool(channel5)

        channel6 = channel6.unsqueeze(1)
        channel6 = F.relu(self.conv1_1(channel6))
        channel6 = F.relu(self.conv1_2(channel6))
        channel6 = self.pool(channel6)

        channel7 = channel7.unsqueeze(1)
        channel7 = F.relu(self.conv1_1(channel7))
        channel7 = F.relu(self.conv1_2(channel7))
        channel7 = self.pool(channel7)

        channel1 = self.adaptive_pool(channel1).squeeze(-1)
        channel2 = self.adaptive_pool(channel2).squeeze(-1)
        channel3 = self.adaptive_pool(channel3).squeeze(-1)
        channel4 = self.adaptive_pool(channel4).squeeze(-1)
        channel5 = self.adaptive_pool(channel5).squeeze(-1)
        channel6 = self.adaptive_pool(channel6).squeeze(-1)
        channel7 = self.adaptive_pool(channel7).squeeze(-1)

        combined = torch.cat((channel1, channel2, channel3, channel4, channel5, channel6, channel7), dim=1)

        combined = torch.flatten(combined, 1)

        combined = F.relu(self.fc1(combined))
        combined = F.relu(self.fc2(combined))
        combined = self.fc3(combined)
        output = combined

        return output.squeeze(-1)

def calculate_mse(model, data_loader, criterion):
    model.eval()  
    total_loss = 0.0
    total_count = 0

    with torch.no_grad():  # Disable gradient computation
        for _, inputs_tuple, labels in data_loader:
            inputs_processed = [inp.to(torch.float32) for inp in inputs_tuple]
            outputs = model(inputs_processed)  
            labels = labels.float() 
            
            loss = criterion(outputs, labels)  
            total_loss += loss.item() * len(labels)  
            total_count += len(labels)  

    mean_mse = total_loss / total_count  
    return mean_mse


if __name__ == "__main__":

    custom_transform = CustomTransform()

    train_dataset = CustomDataset(train_df, transform=custom_transform)
    test_dataset = CustomDataset(test_df, transform=custom_transform)


    batch_size = 40
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    net = Net()

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(net.parameters(), lr=0.00002)

    for epoch in range(100):
        running_loss = 0.0
        print(epoch)
        for i, data in enumerate(train_loader, 0):
            _, inputs_tuple, labels = data


            inputs_processed = []
            for inputs in inputs_tuple:
                inputs = inputs.to(torch.float32)  
                inputs_processed.append(inputs)

            optimizer.zero_grad()

            outputs = net(inputs_processed)  
            labels = labels.float()  

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 5 == 4:
                print(f'[{epoch + 1}, {i + 1:4d}] loss: {running_loss / 5:.3f}')
                running_loss = 0.0

    print('Finished Training')
    

    net.eval()  
    results = []
    with torch.no_grad():
        for identifiers, inputs_tuple, labels in test_loader:
            inputs_processed = [inp.to(torch.float32) for inp in inputs_tuple]
            outputs = net(inputs_processed)


            for id1, id2, id3, id4, id5, id6, id7, id8, output, label in zip(identifiers[0], identifiers[1], identifiers[2], identifiers[3], identifiers[4], identifiers[5], identifiers[6], identifiers[7], outputs, labels):
                deviation = output.item() - label.item()
                results.append([(id1, id2), output.item(), label.item(), deviation])


    results_df = pd.DataFrame(results, columns=['ID', 'Prediction', 'Actual', 'Deviation'])
    results_df.to_csv('RMSprop_test_results.csv', index=False)

    test_mse = calculate_mse(net, test_loader, criterion)
    print(f'Mean Squared Error on Test Set: {test_mse}')


    PATH = './Models/RMSprop_ajusted_net.pth'
    torch.save(net.state_dict(), PATH)
    
    sample_data = next(iter(train_loader))
    sample_inputs = sample_data[1]  

    processed_inputs = [Variable(inp.float()) for inp in sample_inputs]

    output = net(processed_inputs)


    input_dict = {f'input_{i}': inp for i, inp in enumerate(processed_inputs)}
    dot = make_dot(output, params=dict(list(net.named_parameters()) + list(input_dict.items())))
    
    dot.render('./Models/RMSprop_network_graph', format='png')
    


