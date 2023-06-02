import numpy as np
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch import nn
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, f1_score, recall_score

# Load MNIST dataset
print("=> Loading MNIST dataset...")
mnist = fetch_openml('mnist_784', version=1)
inputs = np.array(mnist['data'])
targets = np.array(mnist['target'])

# Visualize a sample digit
sample_digit = inputs[0].reshape(28, 28)
plt.imshow(sample_digit, cmap='binary')
plt.axis('off')
plt.show()

# Preprocess the data

# 1. Standardize inputs
inputs = StandardScaler().fit_transform(inputs) 

# 2. Reshape inputs so that they can be passed to CNN
inputs = torch.from_numpy(inputs)
inputs = torch.reshape(inputs, (inputs.size(0), 1, 28, 28)) # '1' stands for one channel

# 3. Transform labels into One-Hot codes
targets = targets.reshape(targets.shape[0], -1)
targets = OneHotEncoder(sparse=False).fit_transform(targets)
targets = torch.from_numpy(targets)

# 4. Divide the dataset into train_dataset and test_dataset
inputs_train, inputs_test, targets_train, targets_test = train_test_split(inputs, targets, test_size=0.2, random_state=42)


class LeNet(nn.Module):
    
    def __init__(self):
        super(LeNet,self).__init__()
        self.net = nn.Sequential(nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.ReLU(),
                                 nn.AvgPool2d(kernel_size=2, stride=2),
                                 nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(),
                                 nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
                                 nn.Linear(16 * 5 * 5, 10), nn.ReLU())
    
    
    def forward(self, input):
        output = self.net(input)
        return output


# Instantiate a CNN model object 
mymodel = LeNet()
# Determine loss function
loss_fn = nn.CrossEntropyLoss()
# Set optimizer
optimizer = torch.optim.SGD(mymodel.parameters(), lr=0.01, momentum=0.5) 

# Train the model    
print("=> Training...")
batch_size = 2000
epoch = 100
loss = 0
loss_train = []    
for epoch in range(1, (epoch+1)):
    idx = 0
    for batch in range(1, (int)(len(inputs_train)/batch_size)+1):
        predictions_train = mymodel(inputs_train[idx:(idx+batch_size), :, :, :].float())
        loss = loss_fn(predictions_train, targets_train[idx:(idx+batch_size), :])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        idx += batch_size
    loss_train.append(loss.item())

plt.figure()
plt.plot(range(1, (epoch+1)), loss_train, 'r')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Train Loss')
plt.show()

# Evaluate the model on the test set
print("=> Evaluating...")
test_predict_before = mymodel(inputs_test.float())
test_predict_to_numpy = test_predict_before.detach().numpy()
test_target_to_numpy = targets_test.detach().numpy()
test_predict = np.argmax(test_predict_to_numpy, axis=1)
test_target = np.argmax(test_target_to_numpy, axis=1)

# 1、计算混淆矩阵
conf_matrix = confusion_matrix(test_target, test_predict)
conf_matrix = pd.DataFrame(conf_matrix, index=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], 
                           columns=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])  # 数据有10个类别

sns.heatmap(conf_matrix, annot=True)
plt.ylabel('True Labels', fontsize=14)
plt.xlabel('Predictions', fontsize=14)
plt.title('Confusion Matrix')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()

# 2、计算accuracy
print('Accuracy_score: ', accuracy_score(test_target, test_predict))

# 3、计算多分类的precision、recall、f1-score分数
print('Micro precision: ', precision_score(test_target, test_predict, average='micro'))
print('Micro recall: ', recall_score(test_target, test_predict, average='micro'))
print('Micro f1-score: ', f1_score(test_target, test_predict, average='micro'))

print('Macro precision: ', precision_score(test_target, test_predict, average='macro'))
print('Macro recall: ', recall_score(test_target, test_predict, average='macro'))
print('Macro f1-score: ', f1_score(test_target, test_predict, average='macro'))

# 4、显示出每个类别的precision、recall、f1-score
print('Classification_report\n', classification_report(test_target, test_predict))