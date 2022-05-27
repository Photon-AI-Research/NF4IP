# Configuration Reference

## NF4IP
Key | Default | Type | Description
------|-------|-------|-----------
data_dir |  ./data
run |  None | String | name of the run; used to name the checkpoint and tensorboard files
config |  None | String | Additional config file to load. Only available as command-line-option
n_epochs |  50 | Int | Total number of epochs to train
lr |  1e-4 | Float | Learning-Rate
weight_decay | 2e-5 | Float
batch_size |  1600 | Int | Amount of data-points that are processed in parallel
feature |  512 | Int | 
num_blocks |  8 | Int | Number of blocks of the Network
y_noise_scale |  1e-1 | Float |
zeros_noise_scale |  5e-2 | Float |
optimizer | adam | String | Optimizer to use.
retain_graph |  False | Bool | Set True to retain the Graph of the forward step
max_batches_per_epoch |  None | Int | Stop the epoch after specified amount of batches
max_batches_per_validation |  None | Int | only use specified amount of batches for validation
loss_exp_scaling |  False | Bool | Scale loss exponentially
random_seed |  None | Int | if set, it fixes the random seed for reproducible results.
validate_every_epochs |  None | Int | how often to validate. 1 = after every epoch
checkpoint_every_epochs |  5 | Int | how often to checkpoint
overwrite |  False | Bool | overwrite the checkpoint instead of resuming it

## Normalising Flow INN style
Key | Default | Type | Description
------|-------|-------|-----------
enable |  True |  Bool | Enable or disable the module
lambd_predict |  3. | Float |
lambd_latent |  300. | Float |
lambd_rev |  400. | Float |
ndim_pad |  6 | Int | pad the input with zeros
ndim_z |  2 | Int | size of z
inn_network_factory | general_inn | String | Factory of the inn network structure
loss_backward | mdd_multiscale | String | a loss handler
loss_latent | mdd_multiscale | String | a loss handler
loss_fit | mse | String | a loss handler

## VAE
Key | Default | Type | Description
------|-------|-------|-----------
enable | True | Bool | Enable or disable the module
load | None | String | Load VAE model from file

# Handlers
## Losses
Supported losses are currently: mae, mdd_multiscale and mse.

## Optimizers
Supported optimizers are currently: adam, sgd
For a detailed description, see [torch optimizers](https://pytorch.org/docs/stable/optim.html)

### Adam
nf4ip.optimizer = 'adam'

Key | Default | Type | Description
------|-------|-------|-----------
eps | 1e-6 | Float | 
betas | [0.8, 0.9] | List |
amsgrad | False | Bool |

### Sgd
nf4ip.optimizer = 'sgd'

Key | Default | Type | Description
------|-------|-------|-----------
momentum | 0.9 | Float | 

# Hooks and Filters
Note: named parameters are prefixed with an *asterisk
## Cement framework hooks
see [Cement Framework Hooks on cement documentation](https://docs.builtoncement.com/core-foundation/hooks#cement-framework-hooks)

## NF4IP/INN basic Hooks
Hook Name | Description | Parameters
----------|-------------|-----------
pre_training | called just before the training is started | *model(abstractmodel)
post_training | called after the training is finished | *model(abstractmodel)
pre_epoch | called before each epoch | *model(abstractmodel)
post_epoch | called after each epoch | *model(abstractmodel), *i_epoch(int), *n_epochs(int), *loss=(float), *info_dict(dict with additional vars to log)
post_validate | called after validation |  *model(abstractmodel), *i_epoch(int), *n_epochs(int), *loss=(float), *x_samps=(tensor, all data used for validation), *y_samps=(tensor)

## NF4IP/INN basic Filters
Filter Name | Description | Parameters
----------|-------------|-----------
model_parameters | used to collect the model parameters for the optimizer. See VAE for example usage. | trainable_parameters, *model(abstractmodel)
checkpoint_save | called before a checkpoint is saved. allows adding addition data to the checkpoint file | checkpoint(dict)
checkpoint_restore | inverse of checkpoint_save. allows restoration of previously saved data | checkpoint(dict)
train_input | called on every batch | x(tensor), y(tensor)
train_forward_output | called with the forward output | output(tensor)
train_backward_output | called with the backward output | output(tensor)
train_backward_rand_output | called with the backward rand output | output(tensor)
val_input | called on every batch of the validation input | x(tensor), y(tensor)
val_backward_output | called on the validation backward output | output(tensor)