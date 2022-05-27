import torch
def kld_loss(mu, logvar):
    return - 0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
def vae_loss(y_true,y_pred,mu,logvar, beta, gamma):
    mse=torch.nn.MSELoss()
    return mse(y_pred,y_true)+ beta* abs(kld_loss(mu,logvar)-gamma)