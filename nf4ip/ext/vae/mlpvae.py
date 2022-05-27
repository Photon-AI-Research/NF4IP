import torch.nn.functional as F
import torch
import torch.nn as nn

class Config:
    def __init__(self, z, param_count, input_size):
        self.z_dim = z
        self.param_count = param_count
        self.input_size = input_size
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_hidden, lb=0, ub=0, activation = torch.tanh):
        super(MLP, self).__init__()
        torch.manual_seed(1234)
        #self.index = index
        self.linear_layers = nn.ModuleList()
        self.activation = activation
        self.lb = lb 
        self.ub = ub
        self.init_layers(input_size, output_size, hidden_size,num_hidden)
    
    
    def init_layers(self, input_size, output_size, hidden_size, num_hidden):
        torch.manual_seed(1234)
        self.linear_layers.append(nn.Linear(input_size, hidden_size))
        for _ in range(num_hidden):
            self.linear_layers.append(nn.Linear(hidden_size, hidden_size))
        self.linear_layers.append(nn.Linear(hidden_size, output_size))

        for m in self.linear_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
        

    def forward(self, x):
        #x = 2.0 * (x - self.lb) / (self.ub - self.lb) - 1.0
        for i in range(0, len(self.linear_layers) - 1):
            x = self.linear_layers[i](x)
            x = self.activation(x)
        x = self.linear_layers[len(self.linear_layers) - 1](x)
        x = torch.nn.Sigmoid()(x)
        return x

class Reversible(nn.Module):
    def __init__(self, type_ , kernel, ind=1, outd=1, stride=1, pad=0, batch_norm=False, activation_func="relu", input_layer=False, dropout=None):
        super(Reversible, self).__init__()
        self.type = type_
        self.ind = ind
        self.outd = outd
        self.kernel = kernel
        self.stride = stride
        self.pad = pad
        self.use_batch_norm = batch_norm
        self.input_layer = input_layer
        self.dropout = dropout
        
        if activation_func == "relu":
            self.af = nn.ReLU()
        else:
            self.af = None
    
    def get_layers(self, layers):
        return nn.Sequential(*layers)

    def enc(self):
        batch_norm = nn.BatchNorm1d(self.outd)
        if self.type == "conv":
            layer = nn.Conv1d(self.ind, self.outd, self.kernel, padding=self.pad)
        else:
            layer = nn.MaxPool1d(self.kernel)
            self.stride = self.kernel
            
        layers = [layer]
        
        if self.use_batch_norm:
            layers.append(batch_norm)
        if not self.type == "pool":
            if self.dropout:
                layers.append(nn.Dropout(self.dropout))
            layers.append(self.af)
        
        return self.get_layers(layers)
    
    def dec(self):
        batch_norm = nn.BatchNorm1d(self.ind)
        if self.type == "conv":
            layer = nn.ConvTranspose1d(self.outd, self.ind, self.kernel, self.stride, self.pad)
        else:
            layer = nn.ConvTranspose1d(self.ind, self.ind, self.kernel, self.stride, self.pad)
            
        layers = [layer]
        
        if self.use_batch_norm:
            layers.append(batch_norm)
        if not self.input_layer:
            if self.dropout:
                layers.append(nn.Dropout(self.dropout))
            layers.append(self.af)
        
        return self.get_layers(layers)
    

class REncoder(nn.Module):
    def __init__(self, layers):
        super(REncoder, self).__init__()
        self.layers = nn.Sequential(*[layer.enc() for layer in layers])
        
    def forward(self, x):
        return self.layers(x)
    

class RDecoder(nn.Module):
    def __init__(self, layers):
        super(RDecoder, self).__init__()
        self.layers = nn.Sequential(*[layer.dec() for layer in layers[::-1]])
        
    def forward(self, x):
        return torch.nn.Sigmoid()(self.layers(x))
    

class LinearEncoder(nn.Module):
    def __init__(self, outd, z_dim):
        super(LinearEncoder, self).__init__()
        self.fc_mu = nn.Linear(outd, z_dim)
        self.fc_logvar = nn.Linear(outd, z_dim)
        
    def forward(self, x):
        return self.fc_mu(x), self.fc_logvar(x)

class LinearDecoder(nn.Module):
    def __init__(self, outd, z_dim):
        super(LinearDecoder, self).__init__()
        
        self.fc_1 = nn.Linear(z_dim, outd)
        self.fc_2 = nn.Linear(outd, outd)
          
    def forward(self, x):
        x = self.fc_1(x)
        x = F.relu(x)
        x = self.fc_2(x)
        x = F.relu(x)
        return x


class BaseVAE(nn.Module):
    def __init__(self):
        super(BaseVAE, self).__init__()
        
    def flatten(self, x):
        return x.view(x.size(0), -1)
    
    def unflatten(self, x, dims, size):
        return x.view(x.size(0), dims,  size)

    def reparametrize(self, mu, logvar, phase="train"):
        if phase == "train":
            std = logvar.mul(0.5).exp_()
            # return torch.normal(mu, std)
            esp = torch.randn(*mu.size()).cuda()
            z = mu + std * esp
            return z
        else:
            return mu


class ParamGuesser(nn.Module):
    def __init__(self,trained_VAE):
        super(ParamGuesser, self).__init__()
        self.trained_VAE=trained_VAE
        
        self.dense1=nn.Linear(1,20)
        self.dense2=nn.Linear(20,10)
        self.out=nn.Linear(10,3)
    def forward(self,x):
        recon_x, z, mu, logvar, predicted_labels=self.trained_VAE.forward(x)
        x=self.dense1(z)
        x=self.dense2(x)
        x=self.out(x)
        return x


class MLPVAE(BaseVAE):
    def __init__(self, config, mlp_config, cnn_layers):
        super().__init__()

        self.z_dim = config.z_dim
        self.input_size = config.input_size
        self.cnn_dims = int(cnn_layers[-1].outd)
        
        self.encoder = REncoder(cnn_layers)

        def calculate_size():
            x = torch.randn(self.input_size).unsqueeze(0).unsqueeze(0)
            output = self.encoder.layers(x)
            output = torch.flatten(output)
            return output.size()[0]
         
        self.flatten_dim = int(calculate_size())
        self.cnn_size = int(self.flatten_dim/self.cnn_dims)
        self.fc_enc = LinearEncoder(self.flatten_dim, self.z_dim)
        
        self.decoder = RDecoder(cnn_layers)
        self.fc_dec = LinearDecoder(self.flatten_dim, self.z_dim)
        
        #self.mlp = MLP([self.z_dim, 500, 35, config.param_count],config).float()
        self.mlp = MLP(self.z_dim, config.param_count, mlp_config['hidden_size'], mlp_config['num_hidden'], mlp_config['activation']).float()
       
        
    def encode(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        mu, logvar = self.fc_enc(x)
        return mu, logvar
    
    def decode(self, x):
        x = self.fc_dec(x)
        x = self.unflatten(x, self.cnn_dims, self.cnn_size)
        x = self.decoder(x)
        return x

    def deactivate_layers(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def activate_layers(self):
        for param in self.encoder.parameters():
            param.requires_grad = True

    def forward(self, x, phase="train"):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar, phase=phase)
        recon_x = self.decode(z)
        return recon_x, z, mu, logvar
