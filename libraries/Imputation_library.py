from libraries import Transformation_library
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
def impute_one(X_full,X_val, X_part, d, nm, m, seed_sampling): #for imputing tp 0 or tp 1
    X_input = X_part[:, nm, :]
    Y_input = Transformation_library.CLR(X_input, keepdim=False)
    Y_input = Y_input.reshape(-1, 2 * (d-1))
    Y_full, Y_val = Transformation_library.CLR(X_full, keepdim = False), Transformation_library.CLR(X_val, keepdim=False)
    Y_nm, Y_val_nm = Y_full[:, nm, :], Y_val[:, nm, :]  # not missing
    Y_nm, Y_val_nm = Y_nm.reshape(-1, 2 * (d-1)), Y_val_nm.reshape(-1, 2 * (d-1))
    Y_m, Y_val_m = Y_full[:, m, :], Y_val[:, m, :]  # missing
    Y_output = imputation_cGAN0(Y_nm, Y_m, Y_val_nm, Y_val_m, Y_input,  seed_sampling)
    X_part[:, m, :] = Transformation_library.softmax(Y_output)
    return X_part
def impute_two(D_02, D_12, X_val, X_2, nm, m, seed_sampling): #for imputing tp 0 and tp 1
    Y_input = Transformation_library.CLR(X_2[:, nm, :], keepdim = False)
    Y_nm, Y_val_nm = Transformation_library.CLR(D_02[:, nm, :], keepdim = False), Transformation_library.CLR(X_val[:, nm, :], keepdim = False)  # not missing
    Y_m, Y_val_m = Transformation_library.CLR(D_02[:, m[0], :], keepdim = False), Transformation_library.CLR(X_val[:, m[0], :], keepdim = False) # missing
    Y_output = imputation_cGAN0(Y_nm, Y_m, Y_val_nm, Y_val_m, Y_input, seed_sampling)
    X_2[:, m[0], :] = Transformation_library.softmax(Y_output)
    ###
    Y_nm, Y_val_nm = Transformation_library.CLR(D_12[:, nm, :], keepdim = False), Transformation_library.CLR(X_val[:, nm, :], keepdim = False)  # not missing
    Y_m,  Y_val_m= Transformation_library.CLR(D_12[:, m[1], :], keepdim = False), Transformation_library.CLR(X_val[:, m[1], :], keepdim = False)  # missing
    Y_output = imputation_cGAN0(Y_nm, Y_m, Y_val_nm, Y_val_m, Y_input, seed_sampling)
    X_2[:, m[1], :] = Transformation_library.softmax(Y_output)
    return X_2
def imputation_cGAN0(Y_nm, Y_m, Y_val_nm, Y_val_m, Y_input, seed_sampling, latent_dim = 1, epochs = 1000): #imputation by linear regression
    class Discriminator(nn.Module):
        def __init__(self, input_dim):
            super(Discriminator, self).__init__()
            self.disc_logits = nn.Linear(input_dim, 1)
            self.sigmoid = nn.Sigmoid()
        def forward(self, x):
            logits = self.disc_logits(x)
            output = self.sigmoid(logits)
            return output
    class Generator(nn.Module):
        def __init__(self, input_dim, output_dim):
            super(Generator, self).__init__()
            self.gen_output = nn.Linear(input_dim, output_dim)
        def forward(self, x):
            x_hat = self.gen_output(x)
            return x_hat
    class cGAN(nn.Module):
        def __init__(self, Generator, Discriminator):
            super(cGAN, self).__init__()
            self.Generator = Generator
            self.Discriminator = Discriminator
        def forward(self, x, class_dim, target_dim, latent_dim, batch_size):
            x_classes, _ = torch.split(x, [class_dim, target_dim], dim = 1)
            z = torch.randn(batch_size,latent_dim).to(DEVICE)
            latent = torch.concatenate((x_classes, z), dim=1)
            x_hat = self.Generator(latent)
            fake = torch.concatenate((x_classes, x_hat), dim=1)
            discReal = self.Discriminator(x)
            discFake = self.Discriminator(fake)
            return discReal, discFake
    def loss_function(discReal, discFake):
        loss = nn.BCELoss()
        lossDreal = loss(discReal,torch.ones_like(discReal))
        lossDfake = loss(discFake, torch.zeros_like(discFake))
        lossG = loss(discFake, torch.ones_like(discFake))
        return (lossDreal + lossDfake)/2 + lossG
    target_dim = np.shape(Y_m)[1]
    class_dim = np.shape(Y_nm)[1]
    batch_size_train = np.shape(Y_nm)[0]
    batch_size_val = np.shape(Y_val_nm)[0]
    if seed_sampling != None:
        torch.manual_seed(seed_sampling)
    DEVICE = torch.device("cpu")
    gen = Generator(input_dim=class_dim + latent_dim, output_dim=target_dim)
    disc = Discriminator(input_dim=class_dim + target_dim)
    cgan = cGAN(Generator=gen, Discriminator=disc).to(DEVICE)
    optimizer = Adam(cgan.parameters())
    to_stop = 0
    epoch = 0
    last_loss = np.inf
    while epoch < epochs and to_stop == 0:
        epoch += 1
        train_loader = torch.from_numpy(np.concatenate([Y_nm,Y_m],axis=1).astype("float32"))
        val_loader = torch.from_numpy(np.concatenate([Y_val_nm,Y_val_m],axis=1).astype("float32"))
        x = train_loader.to(DEVICE)
        discReal, discFake = cgan(x, class_dim, target_dim, latent_dim,batch_size_train)
        loss = loss_function(discReal, discFake)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        x = val_loader.to(DEVICE)
        discReal, discFake = cgan(x, class_dim, target_dim, latent_dim,batch_size_val)
        val_loss = loss_function(discReal, discFake)
        if val_loss<last_loss:
            last_loss = val_loss
        else:
            to_stop = 1
    print("\t", epoch, val_loss)
    noise = torch.randn(np.shape(Y_input)[0], latent_dim).to(DEVICE)
    Y_input = torch.concatenate((torch.from_numpy(Y_input.astype("float32")), noise), dim=1)
    Y_output = cgan.Generator(Y_input).detach().numpy()
    return Y_output