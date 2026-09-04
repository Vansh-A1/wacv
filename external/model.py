import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.nn.functional as F

kernel_size = 4 # (4, 4) kernel
init_channels = 32 # initial number of filters
image_channels = 3 # MNIST images are grayscale
latent_dim = 64 # latent dimension for sampling

# Define the convolutional variational autoencoder model class
class ConvVAE(nn.Module):
    def __init__(self, latent_dim):
        super(ConvVAE ,self).__init__()
        # Convolutional layers for the encoder
        self.conv1 = nn.Conv2d(3 ,64 ,4 ,2 ,1) # input: 3 x 128 x 128; output: 64 x 64 x 64
        self.conv2 = nn.Conv2d(64 ,128 ,4 ,2 ,1) # input: 64 x 64 x 64; output: 128 x 32 x 32
        self.conv3 = nn.Conv2d(128 ,256 ,4 ,2 ,1) # input: 128 x 32 x 32; output: 256 x 16 x 16
        self.conv4 = nn.Conv2d(256 ,512 ,4 ,2 ,1) # input: 256 x 16 x16; output:512 x8x8

        # Fully connected layers for the mean and log variance of the latent distribution
        self.fc_mean = nn.Linear(512 *8*8 , latent_dim) 
        self.fc_logvar = nn.Linear(512 *8*8 , latent_dim)

        # Fully connected layer for the decoder
        self.fc = nn.Linear(latent_dim, 512 *8*8) # add the one-hot label vector as input

        # Transposed convolutional layers for the decoder
        self.tconv1 = nn.ConvTranspose2d(512 ,256 ,4 ,2 ,1) # input: 512 x 8 x 8; output: 256 x 16 x 16
        self.tconv2 = nn.ConvTranspose2d(256 ,128 ,4 ,2 ,1) # input: 256 x 16 x 16; output: 128 x 32 x 32
        self.tconv3 = nn.ConvTranspose2d(128 ,64 ,4 ,2 ,1) # input: 128 x 32 x32; output:64 x64x64
        self.tconv4 = nn.ConvTranspose2d(64 ,3 ,4 ,2 ,1) # input:64 x64x64; output:3 x128x128

    def encode(self, x):
        # Apply convolutional layers with ReLU activation and batch normalization
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        # Flatten the output of the last convolutional layer
        x = x.view(x.size(0), -1)

        # Compute the mean and log variance of the latent distribution using fully connected layers
        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)

        return mean, logvar

    def reparameterize(self, mean, logvar):
        # Apply the reparameterization trick to sample from the latent distribution
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        return z

    def decode(self, z):
        # Apply the fully connected layer with ReLU activation and batch normalization
        z = F.relu(self.fc(z))

        # Reshape the output of the fully connected layer to match the input of the transposed convolutional layers
        z = z.view(z.size(0), 512, 8, 8)

        # Apply transposed convolutional layers with ReLU activation and batch normalization
        z = F.relu(self.tconv1(z))
        z = F.relu(self.tconv2(z))
        z = F.relu(self.tconv3(z))

        # Apply the last transposed convolutional layer with sigmoid activation to get the reconstructed image
        z = torch.sigmoid(self.tconv4(z))

        return z

    def forward(self, x):
        # Encode the input image and label to get the mean and log variance of the latent distribution
        mean, logvar = self.encode(x)

        # Sample from the latent distribution using the reparameterization trick
        z = self.reparameterize(mean, logvar)

        # Decode the latent vector and label to get the reconstructed image
        recon_x = self.decode(z)

        return recon_x, mean, logvar

    def generate(self, z, y):
      # Decode a given latent vector and label to get an image
      gen_x = self.decode(z, y)
      return gen_x

# Define the convolutional discriminator model class
class ConvDiscriminator(nn.Module):
    def __init__(self):
        super(ConvDiscriminator ,self).__init__()
        # Convolutional layers for the discriminator
        self.conv1 = nn.Conv2d(3 ,64 ,4 ,2 ,1) # input: 3 x 128 x 128; output: 64 x 64 x 64
        self.conv2 = nn.Conv2d(64 ,128 ,4 ,2 ,1) # input: 64 x 64 x 64; output: 128 x 32 x 32
        self.conv3 = nn.Conv2d(128 ,256 ,4 ,2 ,1) # input: 128 x 32 x 32; output: 256 x 16 x 16
        self.conv4 = nn.Conv2d(256 ,512 ,4 ,2 ,1) # input: 256 x 16 x16; output:512 x8x8

        # Fully connected layer for the discriminator output
        self.fc = nn.Linear(512 *8*8, 1) # add the one-hot label vector as input

    def forward(self, x):
        # Apply convolutional layers with LeakyReLU activation and batch normalization
        x = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        x = F.leaky_relu(self.conv2(x), negative_slope=0.2)
        x = F.leaky_relu(self.conv3(x), negative_slope=0.2)
        x = F.leaky_relu(self.conv4(x), negative_slope=0.2)

        # Flatten the output of the last convolutional layer
        x = x.view(x.size(0), -1)

        # Apply the fully connected layer with sigmoid activation to get the discriminator output
        x = torch.sigmoid(self.fc(x))

        return x

class CVAEGenerator(nn.Module):
    def __init__(self):
        super(CVAEGenerator, self).__init__()
 
        # encoder
        self.enc1 = nn.Conv2d(in_channels=image_channels, out_channels=init_channels, kernel_size=kernel_size, stride=2, padding=1)     # input shape 3x512x512 output 64x128x128
        self.enc2 = nn.Conv2d(in_channels=init_channels, out_channels=init_channels*2, kernel_size=kernel_size,stride=2, padding=1)     # input shape 64x128x128 output 64x128x128
        self.enc3 = nn.Conv2d(in_channels=init_channels*2, out_channels=init_channels*4, kernel_size=kernel_size,stride=2, padding=1)   # input shape 64x128x128 output 128x64x64
        self.enc4 = nn.Conv2d(in_channels=init_channels*4, out_channels=init_channels*8, kernel_size=kernel_size,stride=2, padding=1)   # input shape 128x64x64 output 256x32x32
        self.enc5 = nn.Conv2d(in_channels=init_channels*8, out_channels=init_channels*16, kernel_size=kernel_size,stride=2, padding=1)  # input shape 256x32x32 output 512x16x16
        self.enc6 = nn.Conv2d(in_channels=init_channels*16, out_channels=init_channels*32, kernel_size=kernel_size,stride=2, padding=1) # input shape 512x16x16 output 1024x8x8

        # fully connected layers for learning representations
        self.fc1 = nn.Linear(1024, 512)                 # input shape 1024x8x8 output 1024x8x8
        self.fc_mu = nn.Linear(512, latent_dim)         # input shape 512x16x16 output 1024x8x8
        self.fc_log_var = nn.Linear(512, latent_dim)    # input shape 512x16x16 output 1024x8x8
        self.fc2 = nn.Linear(latent_dim, 64)            # input shape 512x16x16 output 1024x8x8
        # decoder 
        self.dec1 = nn.ConvTranspose2d(in_channels=64, out_channels=init_channels*8, kernel_size=kernel_size*16,stride=1, padding=0) 
        self.dec2 = nn.ConvTranspose2d(in_channels=init_channels*8, out_channels=init_channels*4, kernel_size=kernel_size,stride=2, padding=1) 
        self.dec3 = nn.ConvTranspose2d(in_channels=init_channels*4, out_channels=init_channels*2, kernel_size=kernel_size,stride=2, padding=1) 
        self.dec4 = nn.ConvTranspose2d(in_channels=init_channels*2, out_channels=init_channels*2, kernel_size=kernel_size, stride=2, padding=1)
        self.dec5 = nn.ConvTranspose2d(in_channels=init_channels*2, out_channels=image_channels, kernel_size=3,stride=1, padding=1) 
        self.dec6 = nn.ConvTranspose2d(in_channels=init_channels*2, out_channels=image_channels, kernel_size=3, stride=1, padding=1)

    def encode(self, x):
        x1 = F.relu(self.enc1(x))        
        x2 = F.relu(self.enc2(x1))
        x3 = F.relu(self.enc3(x2))
        x4 = F.relu(self.enc4(x3))
        x5 = F.relu(self.enc5(x4))
        x6 = F.relu(self.enc6(x5))
        batch, _, _, _ = x6.shape
        x7 = F.adaptive_avg_pool2d(x6, 1).reshape(batch, -1)
        hidden = self.fc1(x7)

        # get `mu` and `log_var`
        mu = self.fc_mu(hidden)
        log_var = self.fc_log_var(hidden)

        print(f'input shape...{x.shape}')
        print(f'enc1 out shape{x1.shape}')
        print(f'enc2 out shape{x2.shape}')
        print(f'enc3 out shape{x3.shape}')
        print(f'enc4 out shape{x4.shape}')
        print(f'enc5 out shape{x5.shape}')
        print(f'enc6 out shape{x6.shape}')
        print(f'avg pool shape{x7.shape}')
        print(f'hidden shape  {hidden.shape}')
        print(f'mu shape      {mu.shape}')
        print(f'logvar shape  {log_var.shape}')

        return mu, log_var

    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        z = mu + (eps * std) # sampling
        return z
    
    def decode(self, z):
        z = self.fc2(z)
        z1 = z.view(-1, 64, 1, 1)
        # decoding
        x1 = F.relu(self.dec1(z1))
        x2 = F.relu(self.dec2(x1))
        x3 = F.relu(self.dec3(x2))
        x4 = F.relu(self.dec4(x3))

        reconstruction = torch.sigmoid(self.dec5(x4))

        # print(f'decoder input shape...{z.shape}')
        # print(f'z reshape out shape{z1.shape}')
        # print(f'dec1 out shape     {x1.shape}')
        # print(f'dec2 out shape     {x2.shape}')
        # print(f'dec3 out shape     {x3.shape}')
        # print(f'dec4 out shape     {x4.shape}')
        # print(f'recon out shape    {reconstruction.shape}')


        return reconstruction

    def forward(self, x):
        # encoding
        mean, logvar = self.encode(x)
        
        # get the latent vector through reparameterization
        z = self.reparameterize(mean, logvar)

        # decoding
        recon_x = self.decode(z)
        
        return recon_x, mean, logvar

# CVAE with skip connections for generator
class CVAEGenerator_v2(nn.Module):
    def __init__(self, latent_dim=100, image_size=512):
        super(CVAEGenerator_v2, self).__init__()

        self.image_size = image_size

        if self.image_size == 512:
            self.dec1_kernel = 8
            self.dec1_padding = 0
        elif self.image_size == 256:
            self.dec1_kernel = 4
            self.dec1_padding = 0
        elif self.image_size == 224:
            self.dec1_kernel = 4
            self.dec1_padding = 1
        elif self.image_size == 128:
            # encoder is size-agnostic via adaptive pool; only dec1 is size-specific
            self.dec1_kernel = 2
            self.dec1_padding = 0
        else:
            raise ValueError(
                f'CVAEGenerator_v2 supports image_size in {{128,224,256,512}}, got {self.image_size}'
            )

        # encoder
        self.enc1 = nn.Conv2d(in_channels=image_channels, out_channels=init_channels, kernel_size=kernel_size, stride=2, padding=1)     # input shape 3x512x512 output 64x128x128
        self.enc2 = nn.Conv2d(in_channels=init_channels, out_channels=init_channels*2, kernel_size=kernel_size,stride=2, padding=1)     # input shape 64x128x128 output 64x128x128
        self.enc3 = nn.Conv2d(in_channels=init_channels*2, out_channels=init_channels*4, kernel_size=kernel_size,stride=2, padding=1)   # input shape 64x128x128 output 128x64x64
        self.enc4 = nn.Conv2d(in_channels=init_channels*4, out_channels=init_channels*8, kernel_size=kernel_size,stride=2, padding=1)   # input shape 128x64x64 output 256x32x32
        self.enc5 = nn.Conv2d(in_channels=init_channels*8, out_channels=init_channels*16, kernel_size=kernel_size,stride=2, padding=1)  # input shape 256x32x32 output 512x16x16
        self.enc6 = nn.Conv2d(in_channels=init_channels*16, out_channels=init_channels*32, kernel_size=kernel_size,stride=2, padding=1) # input shape 512x16x16 output 1024x8x8

        # fully connected layers for learning representations
        self.fc1 = nn.Linear(1024, 512)                 # input shape 1024x8x8 output 1024x8x8
        self.fc_mu = nn.Linear(512, latent_dim)         # input shape 512x16x16 output 1024x8x8
        self.fc_log_var = nn.Linear(512, latent_dim)    # input shape 512x16x16 output 1024x8x8
        self.fc2 = nn.Linear(latent_dim, 64)            # input shape 512x16x16 output 1024x8x8
        # decoder 
        self.dec1 = nn.ConvTranspose2d(in_channels=64, out_channels=init_channels*8, kernel_size=kernel_size*self.dec1_kernel,stride=1, padding=self.dec1_padding)
        self.dec11 = nn.ConvTranspose2d(in_channels=init_channels*16, out_channels=init_channels*8, kernel_size=3,stride=1, padding=1) 
        
        self.dec2 = nn.ConvTranspose2d(in_channels=init_channels*8, out_channels=init_channels*4, kernel_size=kernel_size,stride=2, padding=1) 
        self.dec3 = nn.ConvTranspose2d(in_channels=init_channels*4, out_channels=init_channels*2, kernel_size=kernel_size,stride=2, padding=1) 
        self.dec4 = nn.ConvTranspose2d(in_channels=init_channels*2, out_channels=init_channels, kernel_size=kernel_size, stride=2, padding=1)
        self.dec5 = nn.ConvTranspose2d(in_channels=init_channels, out_channels=image_channels, kernel_size=kernel_size,stride=2, padding=1) 
        # self.dec6 = nn.ConvTranspose2d(in_channels=init_channels*2, out_channels=image_channels, kernel_size=3, stride=1, padding=1)

    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        z = mu + (eps * std) # sampling
        return z
    
    def forward(self, x):
        # encoding
        x1 = F.relu(self.enc1(x))        
        x2 = F.relu(self.enc2(x1))
        x3 = F.relu(self.enc3(x2))
        x4 = F.relu(self.enc4(x3))
        x5 = F.relu(self.enc5(x4))
        x6 = F.relu(self.enc6(x5))
        batch, _, _, _ = x6.shape
        x7 = F.adaptive_avg_pool2d(x6, 1).reshape(batch, -1)
        hidden = self.fc1(x7)

        # get `mu` and `log_var`
        mu = self.fc_mu(hidden)
        log_var = self.fc_log_var(hidden)
        
        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)

        # print(f'input shape...{x.shape}')
        # print(f'enc1 out shape{x1.shape}')
        # print(f'enc2 out shape{x2.shape}')
        # print(f'enc3 out shape{x3.shape}')
        # print(f'enc4 out shape{x4.shape}')
        # print(f'enc5 out shape{x5.shape}')
        # print(f'enc6 out shape{x6.shape}')
        # print(f'avg pool shape{x7.shape}')
        # print(f'hidden shape  {hidden.shape}')
        # print(f'mu shape      {mu.shape}')
        # print(f'logvar shape  {log_var.shape}')



        # decoding
        z = self.fc2(z)
        # print(f'decoder input shape...{z.shape}')
        z1 = z.view(-1, 64, 1, 1)
        # print(f'z reshape out shape{z1.shape}')
        # decoding
        d1 = F.relu(self.dec1(z1))
        # print(f'dec1 out shape     {d1.shape}')
        d11 = torch.cat([x4, d1], dim=1)
        # print(f'd11 shape          {d11.shape}')

        d1 = F.relu(self.dec11(d11))
        # print(f'd1 shape            {d1.shape}')

        d2 = F.relu(self.dec2(d1))
        # print(f'dec2 out shape     {d2.shape}')
        
        d3 = F.relu(self.dec3(d2))
        # print(f'dec3 out shape     {d3.shape}')
        d4 = F.relu(self.dec4(d3))
        # print(f'dec4 out shape     {d4.shape}')

        # d44 = torch.cat([d4, x1], dim=1)
        # print(f'd44 shape          {d44.shape}')

        # d4 = F.relu(self.dec4(d44))
        # print(f'dec4 out shape     {d4.shape}')

        recon_x = torch.sigmoid(self.dec5(d4))
        # print(f'recon out shape    {recon_x.shape}')


        return recon_x, mu, log_var

class CVAEGenerator512(nn.Module):
    def __init__(self):
        super(CVAEGenerator512, self).__init__()
 
        # encoder
        self.enc1 = nn.Conv2d(in_channels=image_channels, out_channels=init_channels, kernel_size=kernel_size, stride=2, padding=1)
        self.enc2 = nn.Conv2d(in_channels=init_channels, out_channels=init_channels*2, kernel_size=kernel_size,stride=2, padding=1)
        self.enc3 = nn.Conv2d(in_channels=init_channels*2, out_channels=init_channels*4, kernel_size=kernel_size,stride=2, padding=1) 
        self.enc4 = nn.Conv2d(in_channels=init_channels*4, out_channels=init_channels*8, kernel_size=kernel_size,stride=2, padding=1)
        self.enc5 = nn.Conv2d(in_channels=init_channels*8, out_channels=init_channels*16, kernel_size=kernel_size,stride=2, padding=1) 
        self.enc6 = nn.Conv2d(in_channels=init_channels*16, out_channels=init_channels*32, kernel_size=kernel_size,stride=2, padding=1) 

        # fully connected layers for learning representations
        self.fc1 = nn.Linear(1024, 512)
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_log_var = nn.Linear(512, latent_dim)
        self.fc2 = nn.Linear(latent_dim, 64)
        # decoder 
        self.dec1 = nn.ConvTranspose2d(in_channels=64, out_channels=init_channels*8, kernel_size=kernel_size*16,stride=1, padding=0) 
        self.dec2 = nn.ConvTranspose2d(in_channels=init_channels*8, out_channels=init_channels*4, kernel_size=kernel_size,stride=2, padding=1) 
        self.dec3 = nn.ConvTranspose2d(in_channels=init_channels*4, out_channels=init_channels*2, kernel_size=kernel_size,stride=2, padding=1) 
        self.dec4 = nn.ConvTranspose2d(in_channels=init_channels*2, out_channels=init_channels*2, kernel_size=kernel_size, stride=2, padding=1)
        self.dec5 = nn.ConvTranspose2d(in_channels=init_channels*2, out_channels=image_channels, kernel_size=3,stride=1, padding=1) 
        self.dec6 = nn.ConvTranspose2d(in_channels=init_channels*2, out_channels=image_channels, kernel_size=3, stride=1, padding=1)

    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling
        return sample
    
    def forward(self, x):
        # encoding
        # print('input',x.shape)
        x = F.relu(self.enc1(x))
        # print('conv1',x.shape)
        x = F.relu(self.enc2(x))
        # print('conv2',x.shape)
        x = F.relu(self.enc3(x))
        # print('conv3',x.shape)
        x = F.relu(self.enc4(x))
        # print('conv4',x.shape) 
        x = F.relu(self.enc5(x))
        # print('conv4',x.shape)
        x = F.relu(self.enc6(x))
        # print('conv4',x.shape)
        batch, _, _, _ = x.shape
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch, -1)
        # print(x.shape)
        hidden = self.fc1(x)
        # print('hidden', hidden.shape)
        # get `mu` and `log_var`
        mu = self.fc_mu(hidden)
        # print('mu', mu.shape)
        log_var = self.fc_log_var(hidden)
        # print('log_var', log_var.shape)
        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)
        # print('z', z.shape)
        z = self.fc2(z)
        # print('fc2', z.shape)
        z = z.view(-1, 64, 1, 1)
        # print('z', z.shape)
        # decoding
        x = F.relu(self.dec1(z))
        # print('dec1',x.shape)
        x = F.relu(self.dec2(x))
        # print('dec2',x.shape)
        x = F.relu(self.dec3(x))
        # print('dec3',x.shape)
        x = x = F.relu(self.dec4(x))
        # print('dec4',x.shape)
        # x = x = F.relu(self.dec5(x))
        # print('dec5',x.shape)
        reconstruction = torch.sigmoid(self.dec5(x))
        # print('reconstruction',reconstruction.shape)
        return reconstruction, mu, log_var

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 4, 2, 1, bias=False)
        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1, bias=False)
        self.conv3 = nn.Conv2d(128, 256, 4, 2, 1, bias=False)
        self.conv4 = nn.Conv2d(256, 512, 4, 2, 1, bias=False)
        self.conv5 = nn.Conv2d(512, 1, 4, 1, 0, bias=False)
        self.leakyrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.leakyrelu(self.conv1(x))
        x = self.leakyrelu(self.conv2(x))
        x = self.leakyrelu(self.conv3(x))
        x = self.leakyrelu(self.conv4(x))
        x = self.conv5(x)
        return x.view(-1, 1)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
 
        # encoder
        self.enc1 = nn.Conv2d(in_channels=image_channels, out_channels=init_channels, kernel_size=kernel_size, stride=2, padding=1)
        self.enc2 = nn.Conv2d(in_channels=init_channels, out_channels=init_channels*2, kernel_size=kernel_size,stride=2, padding=1)
        self.enc3 = nn.Conv2d(in_channels=init_channels*2, out_channels=init_channels*4, kernel_size=kernel_size,stride=2, padding=1) 
        self.enc4 = nn.Conv2d(in_channels=init_channels*4, out_channels=init_channels*8, kernel_size=kernel_size,stride=2, padding=1)
        self.enc5 = nn.Conv2d(in_channels=init_channels*8, out_channels=init_channels*16, kernel_size=kernel_size,stride=2, padding=1) 
        self.enc6 = nn.Conv2d(in_channels=init_channels*16, out_channels=init_channels*32, kernel_size=kernel_size,stride=2, padding=1) 

        # fully connected layers for learning representations
        self.fc1 = nn.Linear(1024, 512)
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_log_var = nn.Linear(512, latent_dim)

    def forward(self,x):
        x1 = F.relu(self.enc1(x))
        x2 = F.relu(self.enc2(x1))
        x3 = F.relu(self.enc3(x2))
        x4 = F.relu(self.enc4(x3))
        x5 = F.relu(self.enc5(x4))
        x6 = F.relu(self.enc6(x5))
        batch, _, _, _ = x6.shape
        x7 = F.adaptive_avg_pool2d(x6, 1).reshape(batch, -1)
        hidden = self.fc1(x7)

        # get `mu` and `log_var`
        mu = self.fc_mu(hidden)
        log_var = self.fc_log_var(hidden)

        return mu, log_var

# Define the Unet encoder block
class UnetEncoderBlock(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
    super(UnetEncoderBlock, self).__init__()
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
    self.bn = nn.BatchNorm2d(out_channels)
    self.relu = nn.ReLU(inplace=True)

  def forward(self, x):
    x = self.conv(x)
    x = self.bn(x)
    x = self.relu(x)
    return x

# Define the Unet decoder block
class UnetDecoderBlock(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1):
    super(UnetDecoderBlock, self).__init__()
    self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
    self.bn = nn.BatchNorm2d(out_channels)
    self.relu = nn.ReLU(inplace=True)

  def forward(self, x):
    x = self.conv_transpose(x)
    x = self.bn(x)
    x = self.relu(x)
    return x

# Define the variational autoencoder using Unet like architecture
class VAE_UNet(nn.Module):
  def __init__(self, in_channels=3, out_channels=3, latent_dim=128):
    super(VAE_UNet, self).__init__()
    # Define the encoder layers
    self.enc1 = UnetEncoderBlock(in_channels, 64) # output size: (batch_size, 64, h/2, w/2)
    self.enc2 = UnetEncoderBlock(64, 128) # output size: (batch_size, 128, h/4, w/4)
    self.enc3 = UnetEncoderBlock(128, 256) # output size: (batch_size, 256, h/8, w/8)


    # Define the latent layers
    self.fc_mu = nn.Linear(256 * 64 * 64, latent_dim) # output size: (batch_size, latent_dim)
    self.fc_logvar = nn.Linear(256 * 64 * 64, latent_dim) # output size: (batch_size, latent_dim)
    # Define the decoder layers
    self.dec1 = UnetDecoderBlock(latent_dim + 256 + 128 + 64 + in_channels , 256) # output size: (batch_size, 256 , h/4 , w/4 )
    self.dec2 = UnetDecoderBlock(256 , 128) # output size: (batch_size , 128 , h/2 , w/2 )
    self.dec3 = UnetDecoderBlock(128 , out_channels , stride=1 , padding=0 , output_padding=0 ) # output size: (batch_size , out_channels , h , w )

  def reparameterize(self , mu , logvar ):
    # Reparameterize the latent vector using the mean and log variance
    std = torch.exp(0.5 * logvar )
    eps = torch.randn_like(std )
    z = mu + eps * std
    return z

  def forward(self , x ):
    # Encode the input image
    enc1_out = self.enc1(x ) # shape: (batch_size , 64 , h/2 , w/2 )
    enc2_out = self.enc2(enc1_out ) # shape: (batch_size , 128 , h/4 , w/4 )
    enc3_out = self.enc3(enc2_out ) # shape: (batch_size , 256 , h/8 , w/8 )
    print(f'enc1_out {enc1_out.shape}')
    print(f'enc2_out {enc2_out.shape}')
    print(f'enc3_out {enc3_out.shape}')


    # Flatten the encoder output and get the mean and log variance of the latent vector
    enc3_out_flat = enc3_out.view(enc3_out.size(0), -1) # shape: (batch_size , 256 * h/8 * w/8 )
    print(f'enc3_out_flat {enc3_out_flat.shape}')
    
    mu = self.fc_mu(enc3_out_flat ) # shape: (batch_size , latent_dim )
    print(f'mu {mu.shape}')
    logvar = self.fc_logvar(enc3_out_flat ) # shape: (batch_size , latent_dim )
    print(f'logvar {logvar.shape}')

    # Reparameterize the latent vector
    z = self.reparameterize(mu , logvar ) # shape: (batch_size , latent_dim )
    print(f'z {z.shape}')

    # Concatenate the latent vector with the encoder outputs and the input image along the channel dimension
    z_cat = torch.cat([z.unsqueeze(-1).unsqueeze(-1)] * (enc3_out.size(2) * enc3_out.size(3)), dim=1) # shape: (batch_size , latent_dim , h/8 , w/8 )
    print(f'z_cat {z_cat.shape}')
    
    dec1_in = torch.cat([z_cat , enc3_out ], dim=1) # shape: (batch_size , latent_dim + 256 + 128 + 64 + in_channels , h/8 , w/8 )
    print(f'dec1_in {dec1_in.shape}')

    # , F.interpolate(enc2_out , scale_factor=2) , F.interpolate(enc1_out , scale_factor=4) , F.interpolate(x , scale_factor=8)

    # Decode the concatenated vector
    dec1_out = self.dec1(dec1_in ) # shape: (batch_size , 256 , h/4 , w/4 )
    dec2_out = self.dec2(dec1_out ) # shape: (batch_size , 128 , h/2 , w/2 )
    dec3_out = self.dec3(dec2_out ) # shape: (batch_size , out_channels , h , w )

    
    
    
    
    
    
   
   
    print(f'dec1_out {dec1_out.shape}')
    print(f'dec2_out {dec2_out.shape}')
    print(f'dec3_out {dec3_out.shape}')


    # Return the reconstructed image, mean and log variance of the latent vector
    return dec3_out, mu, logvar

# Define the discriminator network for Unet CVAE Generator
class Discriminator_Unet(nn.Module):
  def __init__(self, in_channels=3):
    super(Discriminator_Unet, self).__init__()
    # Define the convolutional layers
    self.conv1 = nn.Conv2d(in_channels, 64, 4, 2, 1)
    self.conv2 = nn.Conv2d(64, 128, 4, 2, 1)
    self.conv3 = nn.Conv2d(128, 256, 4, 2, 1)
    self.conv4 = nn.Conv2d(256, 512, 4, 2, 1)
    self.conv5 = nn.Conv2d(512, 1, 4, 1, 0)
    # Define the batch normalization layers
    self.bn1 = nn.BatchNorm2d(64)
    self.bn2 = nn.BatchNorm2d(128)
    self.bn3 = nn.BatchNorm2d(256)
    self.bn4 = nn.BatchNorm2d(512)
    # Define the leaky ReLU activation function
    self.lrelu = nn.LeakyReLU(0.2)

  def forward(self, x):
    # Apply the convolutional layers with batch normalization and leaky ReLU activation
    x = self.lrelu(self.bn1(self.conv1(x)))
    x = self.lrelu(self.bn2(self.conv2(x)))
    x = self.lrelu(self.bn3(self.conv3(x)))
    x = self.lrelu(self.bn4(self.conv4(x)))
    # Apply the final convolutional layer without batch normalization or activation
    x = self.conv5(x)
    # Return the output
    return x

