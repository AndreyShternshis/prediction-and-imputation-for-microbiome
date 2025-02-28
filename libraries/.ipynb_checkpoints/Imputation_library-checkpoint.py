from libraries import Transformation_library
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression
import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

#The options for imputations are
#imputation_cGAN1(Y_nm, Y_m, Y_input, seed_sampling, Y_val_nm, Y_val_m)
#imputation_CVAE1(Y_nm, Y_m, Y_input, seed_sampling, Y_val_nm, Y_val_m)
#imputation_GPR(Y_nm, Y_m, Y_input, seed_sampling)
#imputation_linear(Y_nm, Y_m, Y_input)
#imputation_SVR(Y_nm, Y_m, Y_input)
def impute(X_full, X_val, X_part, nm, m, seed_sampling, arg): #for imputing only needed bacteria
    X_input = X_part[:, nm, :]
    Y_input = Transformation_library.CLR(X_input, keepdim=False)
    Y_full, Y_val = Transformation_library.CLR(X_full, keepdim = False), Transformation_library.CLR(X_val, keepdim=False)
    Y_nm, Y_val_nm = Y_full[:, nm, :], Y_val[:, nm, :]  # not missing
    Y_m, Y_val_m = Y_full[:, m, arg], Y_val[:, m, arg]  # missing
    Y_output = imputation_SVR(Y_nm, Y_m, Y_input)
    X_part[:, m, arg] = Transformation_library.softmax(Y_output)[:,:-1]
    return X_part

def impute_one(X_full, X_val, X_part, nm, m, seed_sampling):  #for imputing one tp from two tps
    X_input = X_part[:, nm, :]
    Y_input = Transformation_library.CLR(X_input, keepdim=False)
    Y_input = Y_input.reshape(np.shape(Y_input)[0], -1)
    Y_full, Y_val = Transformation_library.CLR(X_full, keepdim=False), Transformation_library.CLR(X_val,keepdim=False)
    Y_nm, Y_val_nm = Y_full[:, nm, :], Y_val[:, nm, :]  # not missing
    Y_nm, Y_val_nm = Y_nm.reshape(np.shape(Y_nm)[0], -1), Y_val_nm.reshape(np.shape(Y_val_nm)[0], -1)
    Y_m, Y_val_m = Y_full[:, m, :], Y_val[:, m, :]  # missing
    Y_output = imputation_SVR(Y_nm, Y_m, Y_input)
    X_part[:, m, :] = Transformation_library.softmax(Y_output)
    return X_part

def impute_two(D_02, D_12, X_val, X_2, nm, m, seed_sampling): #for imputing tp 0 and tp 1
    Y_input = Transformation_library.CLR(X_2[:, nm, :], keepdim = False)
    Y_nm, Y_val_nm = Transformation_library.CLR(D_02[:, nm, :], keepdim = False), Transformation_library.CLR(X_val[:, nm, :], keepdim = False)  # not missing
    Y_m, Y_val_m = Transformation_library.CLR(D_02[:, m[0], :], keepdim = False), Transformation_library.CLR(X_val[:, m[0], :], keepdim = False) # missing
    Y_output = imputation_SVR(Y_nm, Y_m, Y_input)
    X_2[:, m[0], :] = Transformation_library.softmax(Y_output)
    ###
    Y_nm, Y_val_nm = Transformation_library.CLR(D_12[:, nm, :], keepdim = False), Transformation_library.CLR(X_val[:, nm, :], keepdim = False)  # not missing
    Y_m,  Y_val_m= Transformation_library.CLR(D_12[:, m[1], :], keepdim = False), Transformation_library.CLR(X_val[:, m[1], :], keepdim = False)  # missing
    Y_output = imputation_SVR(Y_nm, Y_m, Y_input)
    X_2[:, m[1], :] = Transformation_library.softmax(Y_output)
    return X_2

def imputation_GPR(Y_nm, Y_m, Y_input, seed_sampling, Y_val_nm=[], Y_val_m=[]):
    gp = GaussianProcessRegressor(n_restarts_optimizer=5, random_state = seed_sampling)
    gp.fit(Y_nm, Y_m)
    Y_output = gp.predict(Y_input)
    return Y_output

def imputation_linear(Y_nm, Y_m, Y_input, seed_sampling=10, Y_val_nm=[], Y_val_m=[]):
    reg = LinearRegression()
    reg.fit(Y_nm, Y_m)
    Y_output = reg.predict(Y_input)
    return Y_output

def imputation_SVR(Y_nm, Y_m, Y_input, seed_sampling=10, Y_val_nm=[], Y_val_m=[]):
    Y_output = []
    reg = make_pipeline(StandardScaler(), SVR())
    for i in range(np.shape(Y_m)[1]):
        reg.fit(Y_nm, Y_m[:,i])
        Y_output_line = reg.predict(Y_input).reshape(-1,1)
        Y_output = Y_output_line if np.size(Y_output) == 0 else np.concatenate([Y_output, Y_output_line], axis = -1)
    return Y_output

def imputation_cGAN1(Y_nm, Y_m, Y_input, seed_sampling, Y_val_nm, Y_val_m, latent_dim = 1, epochs = 1000):
    class Discriminator(nn.Module):
        def __init__(self, input_dim, hidden_dim):
            super(Discriminator, self).__init__()
            self.disc_hidden = nn.Linear(input_dim, hidden_dim)
            self.disc_logits = nn.Linear(hidden_dim, 1)
            self.sigmoid = nn.Sigmoid()
            self.tanh = nn.Tanh()
        def forward(self, x):
            hidden = self.tanh(self.disc_hidden(x))
            logits = self.disc_logits(hidden)
            output = self.sigmoid(logits)
            return output
    class Generator(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim):
            super(Generator, self).__init__()
            self.gen_hidden = nn.Linear(input_dim, hidden_dim)
            self.gen_output = nn.Linear(hidden_dim, output_dim)
            self.tanh = nn.Tanh()
        def forward(self, x):
            hidden = self.tanh(self.gen_hidden(x))
            x_hat = self.gen_output(hidden)
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
    gen = Generator(input_dim=class_dim + latent_dim, output_dim=target_dim, hidden_dim=target_dim)
    disc = Discriminator(input_dim=class_dim + target_dim, hidden_dim=target_dim)
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
    #print("\t", epoch, val_loss)
    noise = torch.randn(np.shape(Y_input)[0], latent_dim).to(DEVICE)
    Y_input = torch.concatenate((torch.from_numpy(Y_input.astype("float32")), noise), dim=1)
    Y_output = cgan.Generator(Y_input).detach().numpy()
    return Y_output

def imputation_CVAE1(Y_nm, Y_m, Y_input, seed_sampling, Y_val_nm, Y_val_m, latent_dim = 1, epochs = 1000):
    class Encoder(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim):
            super(Encoder, self).__init__()
            self.encoder_hidden = nn.Linear(input_dim, hidden_dim)
            self.encoder_mean = nn.Linear(hidden_dim, output_dim)
            self.encoder_logvar = nn.Linear(hidden_dim, output_dim)
            self.tanh = nn.Tanh()
        def forward(self, x):
            hidden = self.tanh(self.encoder_hidden(x))
            mean = self.encoder_mean(hidden)
            log_var = self.encoder_logvar(hidden)
            return mean, log_var
    class Decoder(nn.Module):
        def __init__(self, input_dim, output_dim, hidden_dim):
            super(Decoder, self).__init__()
            self.decoder_hidden = nn.Linear(input_dim, hidden_dim)
            self.decoder_logits = nn.Linear(hidden_dim, output_dim)
            self.tanh = nn.Tanh()
        def forward(self, x):
            hidden = self.tanh(self.decoder_hidden(x))
            x_hat = self.decoder_logits(hidden)
            return x_hat
    class CVAE(nn.Module):
        def __init__(self, Encoder, Decoder):
            super(CVAE, self).__init__()
            self.Encoder = Encoder
            self.Decoder = Decoder
        def reparameterization(self, mean, var):
            epsilon = torch.randn_like(var).to(DEVICE)  # sampling epsilon
            z = mean + var * epsilon  # reparameterization trick
            return z
        def forward(self, x, class_dim, target_dim):
            x_classes, x_target = torch.split(x, [class_dim, target_dim], dim = 1)
            mean, log_var = self.Encoder(x)
            z = self.reparameterization(mean, torch.exp(0.5 * log_var))
            latent = torch.concatenate((x_classes, z), dim=1)
            x_hat = self.Decoder(latent)
            return x_target, x_hat, mean, log_var
    def loss_function(x_target, x_hat, mean, log_var):
        reconstruction_loss = torch.mean((x_hat-x_target).pow(2))
        kl_loss = torch.mean(0.5 * torch.sum(mean.pow(2) + log_var.exp() - log_var - 1, dim=1))
        return reconstruction_loss + kl_loss
    target_dim = np.shape(Y_m)[1]
    class_dim = np.shape(Y_nm)[1]
    torch.manual_seed(seed_sampling)
    DEVICE = torch.device("cpu")
    encoder = Encoder(input_dim=class_dim + target_dim, output_dim=latent_dim, hidden_dim=target_dim)
    decoder = Decoder(input_dim=class_dim + latent_dim, output_dim=target_dim, hidden_dim=target_dim)
    cvae = CVAE(Encoder=encoder ,Decoder=decoder).to(DEVICE)
    optimizer = Adam(cvae.parameters())
    to_stop = 0
    epoch = 0
    last_loss = np.inf
    while epoch < epochs and to_stop == 0:
        epoch += 1
        train_loader = torch.from_numpy(np.concatenate([Y_nm,Y_m],axis=1).astype("float32"))
        val_loader = torch.from_numpy(np.concatenate([Y_val_nm,Y_val_m],axis=1).astype("float32"))
        x = train_loader.to(DEVICE)
        x_target, x_hat, mean, log_var = cvae(x, class_dim, target_dim)
        loss = loss_function(x_target, x_hat, mean, log_var)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        x = val_loader.to(DEVICE)
        x_target, x_hat, mean, log_var = cvae(x, class_dim, target_dim)
        loss = loss_function(x_target, x_hat, mean, log_var)
        val_loss = loss.item()
        if val_loss<last_loss:
            last_loss = val_loss
        else:
            to_stop = 1
    #print("\t", epoch, val_loss)
    noise = torch.randn(np.shape(Y_input)[0], latent_dim).to(DEVICE)
    Y_input = torch.concatenate((torch.from_numpy(Y_input.astype("float32")), noise), dim=1)
    Y_output = cvae.Decoder(Y_input).detach().numpy()
    return Y_output