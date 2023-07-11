import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch

from tqdm.auto import tqdm

# training function
def train(model, trainloader, optimizer, criterion, device):
    model.train()
    print("Training")
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    for i, data in tqdm(enumerate(trainloader), total=len(trainloader), position=0, leave=True):
        counter += 1
        image, labels = data
        image = image.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        
        outputs = model(image)
        loss = criterion(outputs, labels)
        train_running_loss += loss.item()

        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == labels).sum().item()

        loss.backward()
        optimizer.step()
    
    epoch_loss = train_running_loss / counter
    epoch_acc = 100 * (train_running_correct / len(trainloader.dataset))
    return epoch_loss, epoch_acc

# validation function
def validate(model, testloader, criterion, device):
    model.eval()
    print("validation")
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0

    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1

            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            
            outputs = model(image)
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()

            _,preds = torch.max(outputs.data, 1)
            valid_running_correct += (preds == labels).sum().item()
        
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100 * (valid_running_correct / len(testloader.dataset))
    return epoch_loss, epoch_acc