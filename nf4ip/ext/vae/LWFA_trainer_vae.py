import torch
import numpy as np
from nf4ip.ext.vae.loss import vae_loss
import torch.optim as optim
import requests


        
class LWFA_Trainer:
    def __init__(self, base_vae, train_loader, val_loader):
        self.base_vae = base_vae
        self.train_loader = train_loader
        self.val_loader = val_loader
        return
    
    def train_VAE(self, lr, epochs, beta, gamma, log_writer=None, send=False):
        lr = lr
        epochs = epochs
        optimizer = optim.Adam(self.base_vae.parameters(), lr=lr)
        loss_fn = torch.nn.MSELoss()
        for k in range(epochs):
            l = []
            for in_net, label in self.train_loader:
                in_net = in_net.to("cuda:0")
                label = label.to("cuda:0")

                optimizer.zero_grad()
                in_net = in_net.unsqueeze(1)
                recon_x, z, mu, logvar = self.base_vae.forward(in_net.float())
                loss = vae_loss(in_net.float(), recon_x, mu, logvar, beta, gamma)
                loss.backward()
                l.append(loss.item())
                optimizer.step()
            if k%5 == 0:
                val = []
                for in_net, label in self.val_loader:
                    in_net = in_net.to("cuda:0")
                    label = label.to("cuda:0")
                    in_net = in_net.unsqueeze(1)  
                    recon_x, z, mu, logvar=self.base_vae.forward(in_net.float())
                    loss= loss_fn(recon_x, in_net.float())
                    val.append(loss.item())
                if log_writer != None:
                    log_writer.add_scalar('val Loss', np.mean(val), k)
                print('[%d] val loss: %f' % (k, np.mean(val)))
            if log_writer != None:
                log_writer.add_scalar('Loss', np.mean(l), k)
            print('[%d] loss: %f' % (k, np.mean(l)))
        return np.mean(val)