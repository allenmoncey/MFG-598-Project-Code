import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from scipy.io import wavfile
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import torchvision.transforms as transforms

batch_size = 64 #batches while training the gan
epochs =75 #epochs to train the GAN
nz = 64 # latent vector size
k = 1 # number of steps to apply to the discriminator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #to use the gpu to train the GAN

#reading the excel file that has all the operations done in exp00 in 0.1 sec intervals
operations = pd.read_excel('Operations001.xlsx') 

# extending the above dataframe as there is 4800 samples in each 0.1 sec
operations4800=pd.DataFrame(np.repeat(operations.values,4800,axis=0)) 

samplerate, x = wavfile.read("EXP00trimmed.wav") #reading the audio recording of the exp00

alloperations=operations4800[0].to_numpy() #transferring to numpy 


'''The for loop below takes only the samples from the audio where facing operation has occured 
and this is done by using the alloperations numpy array which contain the labels of each and 
every sample in the auido file'''

facing=[]
temp=[]
for i in range(len(x)):    
    if alloperations[i]=='Facing Operation ':
        if i>1:
            if alloperations[i-1]=='Facing Operation ':
                temp.append(x[i])
            if alloperations[i-1]!='Facing Operation ':
                facing.append(temp)
                temp=[]
                temp.append(x[i])
           
facing.pop(0)   #removing the first numpy array from facing as it is empty
facing.append(temp) #adding the last temp array to facing


''' The next set of nested for loops creates stft for 0.1 sec intervals and the magnitude of stft is added to the list stfts'''
stfts=[]
z0=[0]*85
for i in range(len(facing)):
    temp=facing[i]
    for j in range(int(len(temp)/4800)):
        y=[]
        y01=temp[j*4800:((j+1)*4800)]
        y.extend(z0)
        y.extend(y01)
        y.extend(z0)
        y.append(0)
        f,t,zxx=signal.stft(y,samplerate,window='hann',nperseg=256,noverlap=128)
        zabs=np.abs(zxx)
        zabs1=zabs[np.newaxis]
        stfts.append(zabs1)


allstft=np.array(stfts) #changed to an numpy array
stfttensor = torch.from_numpy(allstft) #changed to a tensor
stfttensor=stfttensor.float() #changed the datatype to float from double

#creates 10 colour meshes of the real data 
for i in range(10):
    plt.pcolormesh(t, f, stfttensor[i*100][0],cmap='pink')
    plt.title('STFT Magnitude of Real Audio')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.savefig(f"Real STFT {i+1}.png")


mean=torch.mean(stfttensor) #finding the mean for the data
std=torch.std(stfttensor) #finding the standard deviation for the data

normalize = transforms.Normalize(mean=mean, std=std) # Create a normalize transform

stfttensor = normalize(stfttensor) # Normalize a the data

#creating the generator
class Generator(nn.Module):
    def __init__(self, nz):
        super(Generator, self).__init__()
        self.nz = nz
        self.main = nn.Sequential(
            nn.Linear(self.nz, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),

            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),

            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),

            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),

            nn.Linear(1024, 2048),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),

            nn.Linear(2048, 4096),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),

            nn.Linear(4096, 8192),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),

            nn.Linear(8192, 5160),
            nn.Dropout(0.4),

            
        )

    def forward(self, x):
        return self.main(x).view(-1, 1, 129, 40) #changing the tensors array shape

#creating the dicriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.n_input = 5160
        self.main = nn.Sequential(
            nn.Linear(self.n_input, 8192),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),

            nn.Linear(8192, 4096),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),

            nn.Linear(4096, 2048),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),

            nn.Linear(2048, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),

            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),

            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),

            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.view(-1, 5160) #changing the tensors array shape
        return self.main(x)

generator = Generator(nz).to(device) #creating the generator variable
discriminator = Discriminator().to(device) #creating the discriminator variable

print('##### GENERATOR #####')
print(generator)
print('######################')

print('\n##### DISCRIMINATOR #####')
print(discriminator)
print('######################')

# optimizers
optim_g = optim.Adam(generator.parameters(), lr=0.0001)
optim_d = optim.Adam(discriminator.parameters(), lr=0.0001)

# loss function
criterion = nn.BCELoss()

losses_g = [] # to store generator loss after each epoch
losses_d = [] # to store discriminator loss after each epoch
images = [] # to store images generatd by the generator

# to create real labels (1s)
def label_real(size):
    data = torch.ones(size, 1)
    return data.to(device)

# to create fake labels (0s)
def label_fake(size):
    data = torch.zeros(size, 1)
    return data.to(device)

# function to create the noise vector
def create_noise(sample_size, nz):
    return torch.randn(sample_size, nz).to(device)

# to save the images generated by the generator
def save_generator_image(image, path):
    save_image(image, path)

# function to train the discriminator network
def train_discriminator(optimizer, data_real, data_fake):
    b_size = data_real.size(0)
    real_label = label_real(b_size)
    fake_label = label_fake(b_size)

    optimizer.zero_grad()

    output_real = discriminator(data_real)
    loss_real = criterion(output_real, real_label)

    output_fake = discriminator(data_fake)
    loss_fake = criterion(output_fake, fake_label)

    loss_real.backward()
    loss_fake.backward()
    optimizer.step()

    return loss_real + loss_fake

# function to train the generator network
def train_generator(optimizer, data_fake):
    b_size = data_fake.size(0)
    real_label = label_real(b_size)

    optimizer.zero_grad()

    output = discriminator(data_fake)
    loss = criterion(output, real_label)

    loss.backward()
    optimizer.step()

    return loss    



generator.train()
discriminator.train()

#training the GAN

for epoch in range(epochs):
    loss_g = 0.0
    loss_d = 0.0
    for bi in range(0,int(len(stfttensor)/batch_size +1)):
        image = stfttensor[bi*64:(bi+1)*64]
        image = image.to(device)
        b_size = len(image)
        # run the discriminator for k number of steps
        for step in range(k):
            data_fake = generator(create_noise(b_size, nz)).detach()
            data_real = image
            # train the discriminator network
            loss_d += train_discriminator(optim_d, data_real, data_fake)
        data_fake = generator(create_noise(b_size, nz))
        # train the generator network
        loss_g += train_generator(optim_g, data_fake)

    #generating images of the magnitue of stft
    for i in range(10):
        noise1=create_noise(1, nz) #noise tensor
        generated_img = generator(noise1).cpu().detach() #creating the image data using the noise
        generated_img=generated_img*std+mean #reverting the normalization
  
        plt.pcolormesh(t, f, generated_img[0][0],cmap='pink')
        plt.title('STFT Magnitude of Generated Audio')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.savefig(f"Generated STFT {epoch+1} - {i+1}.png")

    epoch_loss_g = loss_g / bi # total generator loss for the epoch
    epoch_loss_d = loss_d / bi # total discriminator loss for the epoch
    losses_g.append(epoch_loss_g)
    losses_d.append(epoch_loss_d)
    
    print(f"Epoch {epoch+1} of {epochs}")
    print(f"Generator loss: {epoch_loss_g:.8f}, Discriminator loss: {epoch_loss_d:.8f}")


print('DONE TRAINING')
torch.save(generator.state_dict(), 'generator.pth') #saving the generator parameters
torch.save(discriminator.state_dict(), 'discriminator.pth') #saving the discriminator parameters

gloss=[]
for i in range(len(losses_g)):
    a=losses_g[i]
    b=a.cpu().detach()
    c=b.numpy()
    gloss.append(c)
gloss
dloss=[]
for i in range(len(losses_d)):
    a=losses_d[i]
    b=a.cpu().detach()
    c=b.numpy()
    dloss.append(c)
dloss
#plotting the loss funtions of the discriminator and the generator
plt.figure()
plt.plot(gloss, label='Generator loss')
plt.plot(dloss, label='Discriminator Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.savefig(f"loss.png")
