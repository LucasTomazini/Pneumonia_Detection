import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import LogSoftmax
from torch import flatten
from torch.optim import Adam
import torch
from torch import nn
import tqdm
import torchvision


np.random.seed(101)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device=',device)

'''
##### Basic Structure #####
# Create a Pytorch Dataset
dataset = CustomDataset(data_dir)

# Create and Train a model
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
model = MyModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

'''



Path = r'.\\chest_xray\\train\\'
Pathv = r'.\\chest_xray\\val\\'

#print(os.listdir(Path))
'''
    transforms.Normalize() é usado para normalizar os canais de cores da imagem. 
    Na chamada transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), 
    os primeiros parâmetros (0.5, 0.5, 0.5) são as médias dos canais vermelho, verde e azul (RGB) respectivamente 
    e os segundos parâmetros (0.5, 0.5, 0.5) são os desvios padrão dos canais RGB, também respectivamente.

    O que essa chamada faz é normalizar cada canal da imagem, subtraindo a média do canal e dividindo pelo desvio padrão. 
    O resultado disso é que a média de cada canal da imagem se torna zero e o desvio padrão se torna 1
'''

class CustomDataset(data.Dataset):
    '''
    Está função converte as imagens e transforma ela em um dataset.
    '''

    def __init__(self, path, transforms=None):
        self.path = path
        self.classes = os.listdir(path)
        self.classes_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        self.img_paths = []
        self.labels = []
        self.transforms = transforms

        for c in self.classes:
            class_dir = os.path.join(self.path, c) # get class directory
            
            for img_name in os.listdir(class_dir): # get images from class directory
                img_path = os.path.join(class_dir, img_name)
                
                self.img_paths.append(img_path)
                self.labels.append(self.classes_to_idx[c])        

    def __getitem__(self, index):
        '''
        Pytorch gives index number automatic.
        '''

        img_path=self.img_paths[index]
        label=self.labels[index]
        img = Image.open(img_path).convert('RGB')

        if self.transforms:
            img = self.transforms(img)
            #print(img)

        return img, label
    
    def __len__(self):

        return len(self.img_paths)


transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
])

dataset = CustomDataset(Path, transforms=transform)

'''
CASO NECESSÁRIO, VERIFICA UMA IMAGEM ALEATÓRIA. JÁ ESTÁ EM 32X32

TRANSF = transforms.ToPILImage()
idx = np.random.randint(0, len(dataset))
img = TRANSF(dataset[idx][0])
img.show()

'''
dataloader = DataLoader(dataset,batch_size=32, shuffle=True)


##### LENET #####
class LeNet(Module):
    def __init__(self, numChannels, classes):
        '''
        numChannels --> 1 grayscale or 3 RGB
        classes --> number of unique classLabels
        '''

        # call the parent constructor
        super(LeNet, self).__init__()

        #initializa the first set of CONV -> ReLu -> Pool Layers
        self.conv1 = Conv2d(in_channels=numChannels, out_channels=6, kernel_size=(5,5))
        self.relu1 = ReLU()
        self.maxpool1 = MaxPool2d(kernel_size=(2,2), stride=(2,2))

        # initialize second set of CONV -: ReLu -> Pool Layers
        self.conv2 = Conv2d(in_channels=6, out_channels=16, kernel_size=(5,5))
        self.relu2 = ReLU()
        self.maxpool2 = MaxPool2d(kernel_size=(2,2), stride=(2,2))

        # initialize set of FC --> ReLU Layers
        self.fc1 = Linear(in_features=16*5*5, out_features=120) #out_channels*kernel
        self.relu3 = ReLU()

        # initialize sofmax classifier
        self.fc2 = Linear(in_features=120, out_features=classes)
        self.logSoftmax = LogSoftmax(dim=1)

    def forward(self, x):

        # applie frist CNN block
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        # applie second CNN block
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        # flatten the output of second block and pass it to thirdth block
        x = flatten(x,1)
        x = self.fc1(x)
        x = self.relu3(x)

        # pass the output of last block to softmax
        x = self.fc2(x)
        output = self.logSoftmax(x)

        return output
    

# define training hyperparaeters

LR = 1e-3
batch_size = 32
epochs = 1

train_split = 0.75
val_split = 1 - train_split

trainSteps = len(dataloader.dataset) // batch_size
print('trainSteps=', trainSteps)

model = LeNet(numChannels=3, classes=2).to(device)

# initialize optimizer Adam and Loss function
opt=Adam(model.parameters(), lr=LR)
lossFn = nn.NLLLoss()

# dictionary to store training store (loss and acc)

History = {
    "train_loss":[],
    "train_Acc":[],
    "val_loss":[],
    "val_acc":[]
}

print("<< INFO >>  trainning the network..")

for epoch in range(epochs):
    print('epoch=', epoch)
    #set model in trainning mode
    model.train()

    # initialize loss
    TotalTrainLoss = 0
    TotalValLoss = 0

    # initialize the number of correct predictions
    TrainCorrect = 0
    ValCorrect = 0

    #looping over trainning set
    for (x,y) in (dataloader):
        
        # send the input to device
        (x,y) = (x.to(device), y.to(device))

        # perform a forward pass and calculate the training loss
        pred = model(x)
        loss = lossFn(pred, y)

        print('LOSS=', loss)

        # zero the gradientes
        # perform backpropagation step
        # update the weights
        opt.zero_grad()
        loss.backward()
        opt.step()

        # add loss to total trinning loss
        # calculate the number of correct predictions

        TotalTrainLoss += loss
        TrainCorrect += (pred.argmax(1)==1).type(torch.float).sum().item()

        History['train_loss'].append(loss)


plt.figure(figsize=(12, 8))

HISTORYY = torch.tensor(History['train_loss']).detach().cpu().numpy()

plt.subplot(2, 2, 1)
#plt.plot(History['loss'], label='Loss')
plt.plot(HISTORYY, label='train_Loss')
#plt.plot(History['train_Acc'], label='train_Acc')

plt.legend()
plt.title('Loss Evolution')

plt.subplot(2, 2, 2)
plt.plot(History['accuracy'], label='Accuracy')
plt.plot(History['val_accuracy'], label='Val_Accuracy')
plt.legend()
plt.title('Accuracy Evolution')
plt.show()

##### FINISH LENET TRAINNING #####

##### TRANSFER LEARNING ON RESNET AND VGG VARIATIONS #####
model_names = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
               'vgg16', 'vgg13', 'vgg16', 'vgg19']

for model_name in model_names:
    #model_name = model_names[0]

    if model_name =='resnet18':
        model = torchvision.models.resnet18(pretrained=True)
    elif model_name =='resnet34':
        model = torchvision.models.resnet34(pretrained=True)
    elif model_name =='resnet50':
        model = torchvision.models.resnet50(pretrained=True)
    elif model_name =='resnet101':
        model = torchvision.models.resnet101(pretrained=True)
    elif model_name =='resnet152':
        model = torchvision.models.resnet152(pretrained=True)
    elif model_name =='vgg11':
        model = torchvision.models.vgg11(pretrained=True)
    elif model_name =='vgg13':
        model = torchvision.models.vgg13(pretrained=True)
    elif model_name =='vgg16':
        model = torchvision.models.vgg16(pretrained=True)
    elif model_name =='vgg19':
        model = torchvision.models.vgg19(pretrained=True)
    else:
        raise ValueError('ModelErrror')


    for param in model.parameters(): # freezing parameters
        param.requires_grad = False

    num_classes = 2
    model.fc = nn.Linear(model.fc.in_features, num_classes) # Makes input layer and output layer suitble to our case.
    model.to(device)

    ResnetTransforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


    Traindataset = CustomDataset(Path, transforms=ResnetTransforms)
    Vldataset = CustomDataset(Pathv, transforms=ResnetTransforms)

    Traindataloader = DataLoader(dataset,batch_size=32, shuffle=True)
    Valdataloader = DataLoader(dataset,batch_size=32, shuffle=True)


    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)

    epochs = 2

    for epoch in range(epochs):
        train_loss, train_correct, val_loss, val_correct = 0, 0, 0, 0

        model.train()
        for x, y in tqdm.tqdm(Traindataloader):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            _, preds = torch.max(outputs, 1)
            train_loss += loss.item()
            train_correct += torch.sum(preds == y.data)
            if epoch % 1000 == 0.0:
                print('{} EPOCH={} LOSS={}'.format(model_name, epoch, loss.item()))

            # validation

            model.eval()
            for x, y in Valdataloader:
                x = x.to(device)
                y = y.to(device)
                outputs = model(x)
                loss = criterion(outputs, y)
                _, preds = torch.max(outputs, 1)
                val_loss += loss.item()
                val_correct += torch.sum(preds == y.data)


            train_loss /= len(Traindataloader)
            train_acc = train_correct.double() / len(Traindataloader)
            val_loss /= len(Valdataloader)
            val_acc = val_correct.double() / len(Valdataloader)

            print("Epoch: {}  Train Loss: {:.4f}  Train Acc: {:.4f}  Val Loss: {:.4f}  Val Acc: {:.4f}".format(
        epoch+1, train_loss, train_acc, val_loss, val_acc))

# FINISH
