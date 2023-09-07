import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Load the CSV file into a Pandas DataFrame
data = pd.read_csv('/Users/abhinavgoel/chess4brainhealth/archive-3/sleep_data.csv')

# Data Preprocessing
# Assume you have already cleaned and encoded the 'Category' column

# Calculate the average 'Category' and 'Heart Rate'
average_category = data['Category'].mode().values[0]  # Most common category
average_heart_rate = round(data['Heart Rate'].mean(), 2)  # Average heart rate rounded to the nearest hundredth

# Fill any missing values in 'Category' and 'Heart Rate' with the calculated averages
data['Category'].fillna(average_category, inplace=True)
data['Heart Rate'].fillna(average_heart_rate, inplace=True)

# Encode the 'Category' column to numerical values
label_encoder = LabelEncoder()
data['Category'] = label_encoder.fit_transform(data['Category'])

# Normalize the 'Heart Rate' column
scaler = MinMaxScaler()
data['Heart Rate'] = scaler.fit_transform(data['Heart Rate'].values.reshape(-1, 1))

# Create input features (X) and presume everyone is healthy (y=1)
X = data[['Category', 'Heart Rate']].values
y = np.ones(X.shape[0])  # Presume everyone is healthy

# Split the dataset into training and testing subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the neural network model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 64)  # 2 input features
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

model = Net()

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Convert data to PyTorch tensors
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train)
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test)

# Training loop
for epoch in range(10):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train.view(-1, 1))
    loss.backward()
    optimizer.step()

# Evaluation
with torch.no_grad():
    outputs = model(X_test)
    predicted = (outputs >= 0.5).float()
    accuracy = (predicted == y_test.view(-1, 1)).sum().item() / len(y_test)
    print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Randomly select an index and extract the corresponding row
random_index = np.random.randint(0, len(data))
random_row = data.iloc[random_index]
original_category = label_encoder.classes_[random_row['Category']]  # Retrieve the original category
original_heart_rate = random_row['Heart Rate'] * scaler.data_max_[0]  # Multiply by max heart rate value

print(f"Random Row - Category: {original_category}, Heart Rate: {original_heart_rate:.2f}")

# Convert the random row to a PyTorch tensor for prediction
random_data = torch.FloatTensor([[random_row['Category'], random_row['Heart Rate']]])
probability = model(random_data).item()
print(f'Likelihood of being healthy: {probability * 100:.2f}%')
