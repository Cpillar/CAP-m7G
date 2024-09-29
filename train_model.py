'''

4-fold CV
3/4 of training samples are used training
1/4 of training set is are used for vaildation.

'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import Adam
from torchvision import datasets, transforms
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score
from sklearn.metrics import auc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report,matthews_corrcoef,f1_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset
import model
import numpy as np
import pandas as pd
import copy
USE_CUDA = True


'''
read CGR images
'''

batch_size = 64
n_epochs = 20
res = 64
fig = pd.read_csv("train.txt")

str_list=[]
for i in fig['figure']:
  temp=np.array(i.split(" "),dtype=np.float32).reshape(res,res)
  str_list.append(temp)
y=fig['label']

X=torch.tensor(str_list)

X=X.unsqueeze(1)

'''
create dataset
'''

dataset=TensorDataset(X,torch.tensor(y))
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

num_cap_list=[4,8,16,32] #find the best num_cap parameter


'''
4-fold CV
'''

kf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
for num_cap in num_cap_list:
    for fold, (train_index, val_index) in enumerate(kf.split(X, y)):
        train_dataset = torch.utils.data.Subset(dataset, train_index)
        val_dataset = torch.utils.data.Subset(dataset, val_index)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) #
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        capsule_net = model.CapsNet(Primary_capsule_num=num_cap)
        if USE_CUDA:
            capsule_net = capsule_net.cuda()
        optimizer = Adam(capsule_net.parameters(), lr=1e-3, betas=(0.9, 0.999))
        best_acc = 0

        for epoch in range(n_epochs):
            capsule_net.train()
            train_loss = 0
            correct_train = 0
            TP_train, FN_train, FP_train = 0, 0, 0

            for batch_id, (data, target) in enumerate(train_loader):
                target = torch.sparse.torch.eye(2).index_select(dim=0, index=target.long())
                data, target = Variable(data), Variable(target)

                if USE_CUDA:
                    data, target = data.cuda(), target.cuda()

                optimizer.zero_grad()
                output, reconstructions, masked = capsule_net(data)
                loss = capsule_net.loss(data, output, target, reconstructions)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            print(f"Train Loss: {avg_train_loss:.4f}" )

            capsule_net.eval()
            with torch.inference_mode():
                correct_val = 0
                TP_val = 0
                FN_val = 0
                FP_val = 0
                best_acc = 0
                best_validation_model = None
                for data, target in val_loader:
                    target = torch.sparse.torch.eye(2).index_select(dim=0, index=target.long())
                    if USE_CUDA:
                        data, target = data.cuda(), target.cuda()
                    output, reconstructions, masked = capsule_net(data)

                    pred_labels = np.argmax(masked.data.cpu().numpy(), 1)
                    true_labels = np.argmax(target.data.cpu().numpy(), 1)
                    correct_val += np.sum(pred_labels == true_labels)
                    TP_val += np.sum((pred_labels == 1) & (true_labels == 1))
                    FN_val += np.sum((pred_labels == 0) & (true_labels == 1))
                    FP_val += np.sum((pred_labels == 1) & (true_labels == 0))

                avg_val_accuracy = correct_val / len(val_loader.dataset)
                recall_val = TP_val / (TP_val + FN_val)
                precision_val = TP_val / (TP_val + FP_val)

                print(f"Fold {fold + 1}, Epoch {epoch + 1}/{n_epochs}")
                print(
                    f"val Accuracy: {avg_val_accuracy:.4f}, val Recall: {recall_val:.4f}, val Precision: {precision_val:.4f}")
                '''
                save the best check point for validation set, and use evaluate code for test set to get the evaluation result.
                '''
                if best_acc < avg_val_accuracy:
                    best_acc = avg_val_accuracy
                    best_validation_model = copy.deepcopy(capsule_net)
                torch.save(best_validation_model, f'{num_cap}_{fold + 1}_{best_acc:.4f}_validation_capsule_net.pth')
