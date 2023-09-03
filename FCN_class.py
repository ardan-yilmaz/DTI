import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.svm import SVC


from utils import plot_loss_curves


class FCN(nn.Module):
    def __init__(self, num_classes, dropout_prob=0.3):
        super(FCN, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(1024, 512), 
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            
        )

        self.classifer = nn.Sequential(
            nn.Linear(128, num_classes)
        )

        self.SVM = None

    def forward(self, x):
        x = self.fc(x)
        x = self.classifer(x)
        return x


    def pre_train_model(self, train_loader, val_loader, lr=1.e-4, epochs=50, patience=10):
        train_losses = []
        val_losses = []

        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss() 

        best_val_loss = float('inf')
        counter = 0

        for epoch in range(epochs):
            self.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            # Validation Loss
            self.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    outputs = self(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

            train_loss = running_loss/len(train_loader)
            val_loss = val_loss/len(val_loader)
            # print losses
            print(f'Epoch: {epoch+1}/{epochs} | Train Loss: {train_loss} | Val Loss: {val_loss}')
            # save losses
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            # Early stopping logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0  # reset the counter
            else:
                counter += 1

            if counter >= patience:
                print(f"Early Stopping at epoch {epoch+1}!")
                break

        
        plot_loss_curves(train_losses, val_losses, 'baseline_MLP_')
        print('Finished Training')

    def extract_features(self, data_loader):
        features = []
        labels = []
        self.eval()
        with torch.no_grad():
            for inputs, label in data_loader:
                feature = self.fc(inputs)
                features.append(feature.cpu().numpy())
                labels.append(label.cpu().numpy())
        return np.vstack(features), np.hstack(labels)    
    


    def fine_tune(self, mode, data_loader, lr=1.e-5, epochs=10):
        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss() 

        if mode == 'full_fine_tune':

            # replace the classifier for binary classification
            self.classifer = nn.Sequential(
                nn.Linear(128, 6)
            )

            # finetune the whole model
            for param in self.parameters():
                param.requires_grad = True

            for epoch in range(epochs):
                self.train()
                running_loss = 0.0
                for inputs, labels in data_loader:
                    optimizer.zero_grad()
                    outputs = self(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()     
                print(f'Epoch: {epoch+1}/{epochs} | Train Loss: {running_loss/len(data_loader)}')           

 
            
        if mode == 'feature_transformer':

            # replace the classifier for binary classification
            self.classifer = nn.Sequential(
                nn.Linear(128, 6)
            )

            # freeze the bottom layers
            for param in self.parameters():
                param.requires_grad = False

            # unfreeze the last layer
            for param in self.fc[-1].parameters():
                param.requires_grad = True

            for epoch in range(epochs):
                self.train()
                running_loss = 0.0
                for inputs, labels in data_loader:
                    optimizer.zero_grad()
                    outputs = self(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()     
                print(f'Epoch: {epoch+1}/{epochs} | Train Loss: {running_loss/len(data_loader)}')


        if mode == 'shallow_classifier':
            # extract features using self.fc
            features, labels = self.extract_features(data_loader)
            # train SVM
            self.SVM = SVC(decision_function_shape='ovr')
            self.SVM.fit(features, labels)    

    
    def get_preds(self, dataloader):
        """
        Used for testing both mutli-class and binary classification
        Should handle all test cases: using an SVM, binary, and mulit-class classification
        """
        if self.SVM is not None:
            # extract features 
            features, labels = self.extract_features(dataloader)

            # get preds using SVM
            preds = self.SVM.predict(features)

            return preds, labels
        
        else:
            preds = []
            labels = []
            self.eval()
            with torch.no_grad():
                for inputs, label in dataloader:
                    outputs = self(inputs)
                    
                    if outputs.shape[1] == 1:  # Binary classification case
                        probabilities = torch.sigmoid(outputs)
                        predicted = (probabilities > 0.5).long().squeeze()  # Thresholding
                    else:  # Multi-class classification case
                        _, predicted = torch.max(outputs.data, 1)
                    
                    preds.extend(predicted.cpu().numpy())
                    labels.extend(label.cpu().numpy())
            
            return np.array(preds), np.array(labels)

            



