from skimage.color import rgb2gray
import scipy
from skimage import filters
from scipy import ndimage
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import VQGAN
import os
import argparse
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch
from torch.distributions import Bernoulli
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
# Set random seed for reproducibility
SEED = 87
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)



def print_model(encoder, decoder):
    print("============== Encoder ==============")
    print(encoder)
    print("============== Decoder ==============")
    print(decoder)
    print("")
 
def create_model(Model_Type):
    if Model_Type == "AE":
        autoencoder = Autoencoder.Autoencoder()  
        print_model(autoencoder.encoder, autoencoder.decoder)
    elif Model_Type == "VQVAE":
        autoencoder = VQGAN.VQGAN()
        print_model(autoencoder.encoder, autoencoder.decoder) 

    if torch.cuda.is_available():
        autoencoder = autoencoder.cuda()
        print("Model moved to GPU in order to speed up training.")
    return autoencoder

def get_torch_vars(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def imshow(img):
    npimg = img.cpu().numpy()
    plt.axis('off')
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()  

def MAE_(fake,real):
    mae = 0.0
    mae = np.mean(np.abs(fake-real))
    return mae


Model_Type = "VQVAE"
autoencoder = create_model(Model_Type)

criterion = nn.BCELoss()
criterion_mse = nn.MSELoss()  
optimizer = optim.Adam(autoencoder.parameters(),lr=0.001) 
latent_loss_weight = 0.25 
total_steps = 100
global_mae = 100000000000000
Exp_ID = "Pretraining_64"
device = "cuda"
Image_size = 64

# 定义transform来resize图像到64x64    
transform = transforms.Compose([
    transforms.Resize(Image_size),
    transforms.ToTensor(),
])
# 加载MNIST数据集
train_set = torchvision.datasets.MNIST(root='../data', train=True, download=True, transform=transform)
test_set = torchvision.datasets.MNIST(root='../data', train=False, download=True, transform=transform)
 
# 加载数据加载器
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False) 

for epoch in range(total_steps):
    running_loss = 0.0
    # 示例：打印一些训练数据
    #Training
    for i, (inputs, _) in enumerate(train_loader, 0): 
        inputs = get_torch_vars(inputs) 

        # ============ Forward ============
        if Model_Type == "AE":
            encoded, outputs = autoencoder(inputs)
            loss = criterion(outputs, inputs)

        elif Model_Type == "VQVAE": 
            #out,_, latent_loss,OCT_Fea,OCTA_QuanFea = autoencoder(inputs)
            outputs, codebook_indices, latent_loss  = autoencoder(inputs) 
            loss =  criterion(outputs, inputs) + latent_loss * latent_loss_weight   
             
        # ============ Backward ============     
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 

        # ============ Logging ============   
        running_loss += loss.data
        if i % 20 == 19:
            print('[%d, %10d]  loss: %.10f' %
                    (epoch + 1, i + 1, running_loss / 20))
            running_loss = 0.0

    #Test
    with torch.no_grad():
        MAE = 0 
        num = 0  
        for i, (inputs, _) in enumerate(test_loader, 0): 
            inputs = get_torch_vars(inputs)  
            #print(inputs)
            # ============ Forward ============
            if Model_Type == "AE":
                encoded, outputs = autoencoder(inputs)
                loss = criterion(outputs, inputs)

            elif Model_Type == "VQVAE": 
                #out,_, latent_loss,OCT_Fea,OCTA_QuanFea = autoencoder(inputs)
                outputs, codebook_indices, latent_loss  = autoencoder(inputs) 
                #mse =  criterion_mse(outputs, inputs) 

                mae = MAE_(outputs.detach().cpu().numpy(),inputs.cpu().numpy())
                MAE += mae  
                num += 1 #len(inputs)

        print ('Test MAE:',MAE/num)
        if MAE/num < global_mae:
            global_mae = MAE/num
            # Save best models checkpoints
            print('saving the current best model at the end of epoch %d, iters %d' % (epoch, total_steps)) 

            save_dir = os.path.join("../Weight", Exp_ID) 
            if not os.path.exists(save_dir): 
                os.makedirs(save_dir)

            save_filename = '%s_net.pth' % (Exp_ID)
            save_filename_encoder = '%s_encoder.pth' % (Exp_ID)
            save_filename_decoder = '%s_decoder.pth' % (Exp_ID)
            save_filename_codebook = '%s_codebook.pth' % (Exp_ID)

            save_path = os.path.join(save_dir, save_filename)
            save_path_encoder  = os.path.join(save_dir, save_filename_encoder)
            save_path_deconder = os.path.join(save_dir, save_filename_decoder)
            save_path_codebook = os.path.join(save_dir, save_filename_codebook)

            torch.save(autoencoder.encoder.cpu().state_dict(), save_path_encoder)
            torch.save(autoencoder.decoder.cpu().state_dict(), save_path_deconder)
            torch.save(autoencoder.codebooks.cpu().state_dict(), save_path_codebook)
            torch.save(autoencoder.cpu().state_dict(), save_path)
            autoencoder.to(device)  
            print("saving best...")



# print('Finished Training')
# print('Saving Model...')
# if not os.path.exists('./weights'): 
#     os.mkdir('./weights')
# torch.save(autoencoder.state_dict(), "./weights/autoencoder.pkl")

