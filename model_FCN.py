import torch
import torch.nn as nn
import torch.optim as optim

from utils import plot_loss_curve

class FCN(nn.Module):
    def __init__(self, num_classes, device):
        super(FCN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(1024, 512), 
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes) 
        )
        self.device = device

        self.shallow_classifier = None

    def forward(self, x):
        x = self.fc(x)
        return x

    def train_model(self, train_dataloader, val_dataloader, optimizer, criterion, num_epochs, to_val=1, to_plot=1):
        self.to(self.device)
        train_losses = []
        val_losses = []
        for epoch in range(num_epochs):
            # Training
            self.train()
            running_loss = 0.0
            for inputs, labels in train_dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            train_loss = running_loss/len(train_dataloader)
            train_losses.append(train_loss)
            
            if to_val:
                # Validation
                self.eval()
                running_val_loss = 0.0
                with torch.no_grad():
                    for inputs, labels in val_dataloader:
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        outputs = self(inputs)
                        loss = criterion(outputs, labels)
                        running_val_loss += loss.item()
                val_loss = running_val_loss/len(val_dataloader)
                val_losses.append(val_loss)

                print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}") 

            else: print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.4f}")
                
            
        if to_plot == 1:  
            plot_loss_curve(train_losses, "train")
            if to_val == 1:
                plot_loss_curve(val_losses, "val")


    


def fine_tune_FCN(classifier, strategy, train_dataloader, val_dataloader, optimizer, criterion, num_epochs):
    
    # add another class for the last layer
    classifier.fc[-1] = nn.Linear(128, 6)

    # check strategy
    if strategy ==  "finetune_last_layer":
        # freeze all layers except the last one
        for param in classifier.parameters():
            param.requires_grad = False
        for param in classifier.fc[-1].parameters():
            param.requires_grad = True   
    
    elif strategy == "finetune_last_two":
        # Freeze all layers first
        for param in classifier.parameters():
            param.requires_grad = False
        # Unfreeze the last two nn.Linear layers
        for param in classifier.fc[-1].parameters():
            param.requires_grad = True
        for param in classifier.fc[-3].parameters():
            param.requires_grad = True 
               
    elif strategy == "finetune_all":
        for param in classifier.parameters():
            param.requires_grad = True    

    
    else:
        print("Invalid strategy. Please choose from 'finetune_last_layer', 'finetune_last_two', 'finetune_all'")
    

    #train_model(self, train_dataloader, val_dataloader, optimizer, criterion, num_epochs, to_val)
    classifier.train_model(train_dataloader, val_dataloader, optimizer, criterion, num_epochs, to_plot=0) 

    return classifier
    

     

