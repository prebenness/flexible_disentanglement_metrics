import copy
import math

import numpy as np
import torch
from metrics.IoB_models import TensorReconstructor, VectorReconstructor
import os
import argparse

from torchsummary import summary

#Load the data directory and saving path
parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, required=True, help='Path to the data file.')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use.')
parser.add_argument('--save', type=str, default='output', help='Path to save the results.')
opts = parser.parse_args()

num_epochs = 500
batch_size = 10
early_stop_patience = 25

gpu_num = opts.gpu
dir_root = os.path.abspath(opts.root)

device = torch.device("cuda:"+gpu_num if torch.cuda.is_available() else "cpu")
train_cont_root = 'content_train.npz'
train_sty_root = 'style_train.npz'
train_img_root = 'images_train.npz'
test_cont_root = 'content_test.npz'
test_sty_root = 'style_test.npz'
test_img_root = 'images_test.npz'

result_dir = os.path.abspath(opts.save)
os.makedirs(result_dir, exist_ok=True)
result_file = os.path.join(result_dir, 'IoB_result.txt')

#Load the data to train IoB models
train_content = np.load(os.path.join(dir_root, train_cont_root))['arr_0']
train_style = np.load(os.path.join(dir_root, train_sty_root))['arr_0']
train_images = np.load(os.path.join(dir_root, train_img_root))['arr_0']
test_content = np.load(os.path.join(dir_root, test_cont_root))['arr_0']
test_style = np.load(os.path.join(dir_root, test_sty_root))['arr_0']
test_images = np.load(os.path.join(dir_root, test_img_root))['arr_0']

train_num_samples = train_images.shape[0]
test_num_samples = test_images.shape[0]

# Transpose data for pytorch
train_content = torch.from_numpy(train_content)
train_style = torch.from_numpy(train_style)
train_images = torch.from_numpy(train_images)
train_content_bias = torch.ones_like(train_content)
train_style_bias = torch.ones_like(train_style)

test_content = torch.from_numpy(test_content)
test_style = torch.from_numpy(test_style)
test_images = torch.from_numpy(test_images)
test_content_bias = torch.ones_like(test_content)
test_style_bias = torch.ones_like(test_style)


def train_tensor_reconstructor(repr, target):
    reconstructor = TensorReconstructor()
    reconstructor.to(device)

    # Create train val split
    repr_train, repr_val, target_train, target_val = train_val_split(repr, target)

    num_itr_train = math.ceil(repr_train.shape[0] / batch_size)

    best_val_loss, num_no_improve = torch.inf, 0
    best_state_dict, best_epoch = copy.deepcopy(reconstructor.state_dict()), 0
    for epoch in range(num_epochs):
        index = torch.randperm(repr_train.shape[0])
        for i in range(num_itr_train):
            repr_batch = repr_train[index[i * batch_size:(i+1)*batch_size], :, :, :].float().to(device)
            target_batch = target_train[index[i * batch_size:(i+1)*batch_size], :, :, :].float().to(device)
            mse_loss = reconstructor.AE_update(repr_batch, target_batch)
        print(f'Epoch {epoch}, MSE: {mse_loss}')

        val_mse_loss = test_tensor_reconstructor(reconstructor, repr_val, target_val)
        print(f'Epoch {epoch}, MSE: {mse_loss}, val MSE: {val_mse_loss}')

        # Early stopping criterium
        if val_mse_loss < best_val_loss:
            best_val_loss = val_mse_loss
            best_state_dict = copy.deepcopy(reconstructor.state_dict())
            best_epoch = epoch
            num_no_improve = 0
        else:
            num_no_improve += 1

        if num_no_improve >= early_stop_patience:
            print(f'Validation loss stopped improving, stopping early and restoring best model from epoch {best_epoch}')
            break

    best_model = VectorReconstructor(input_dim=repr.shape[-1], output_dim=target.shape[-1])
    best_model.load_state_dict(best_state_dict)
    return best_model


def train_vector_reconstructor(repr, target):
    reconstructor = VectorReconstructor(input_dim=repr.shape[-1], output_dim=target.shape[-1])
    reconstructor.to(device)

    # Create train val split
    repr_train, repr_val, target_train, target_val = train_val_split(repr, target)

    num_itr_train = math.ceil(repr_train.shape[0] / batch_size)

    summary(reconstructor, (repr.shape[-1],))

    best_val_loss, num_no_improve = torch.inf, 0
    best_state_dict, best_epoch = copy.deepcopy(reconstructor.state_dict()), 0
    for epoch in range(num_epochs):
        index = torch.randperm(repr_train.shape[0])
        for i in range(num_itr_train):
            repr_batch = repr_train[index[i * batch_size:(i+1)*batch_size], :].float().to(device)
            target_batch = target_train[index[i * batch_size:(i+1)*batch_size], :, :, :].float().to(device)
            mse_loss = reconstructor.DE_update(repr_batch, target_batch)
        
        val_mse_loss = test_vector_reconstructor(reconstructor, repr_val, target_val)
        print(f'Epoch {epoch}, MSE: {mse_loss}, val MSE: {val_mse_loss}')
        
        # Early stopping criterium
        if val_mse_loss < best_val_loss:
            best_val_loss = val_mse_loss
            best_state_dict = copy.deepcopy(reconstructor.state_dict())
            best_epoch = epoch
            num_no_improve = 0
        else:
            num_no_improve += 1

        if num_no_improve >= early_stop_patience:
            print(f'Validation loss stopped improving, stopping early and restoring best model from epoch {best_epoch}')
            break
    
    best_model = VectorReconstructor(input_dim=repr.shape[-1], output_dim=target.shape[-1])
    best_model.load_state_dict(best_state_dict)
    return best_model


def test_tensor_reconstructor(model, repr, target):
    with torch.no_grad():
        model.to(device)
        mse_tot = 0
        for i in range(target.shape[0]):
            repr_batch = repr[i:i+1, :, :, :].float().to(device)
            target_batch = target[i:i+1, :, :, :].float().to(device)
            mse_loss = model.test(repr_batch, target_batch)   
            mse_tot += mse_loss

        mse_av = mse_tot / target.shape[0]

    return mse_av


def test_vector_reconstructor(model, repr, target):
    with torch.no_grad():
        model.to(device)
        mse_tot = 0
        for i in range(target.shape[0]):
            repr_batch = repr[i:i+1, :].float().to(device)
            target_batch = target[i:i+1, :, :, :].float().to(device)
            mse_loss = model.test(repr_batch, target_batch)   
            mse_tot += mse_loss

        mse_av = mse_tot / target.shape[0]

    return mse_av


def train_val_split(t1, t2, train_frac=0.8):
    assert t1.shape[0] == t2.shape[0]

    n = t1.shape[0]
    n_train = math.floor(n * train_frac)

    tv_indeces = torch.randperm(n)
    train_indeces, val_indeces = tv_indeces[:n_train], tv_indeces[n_train:]

    t1_train, t2_train = t1[train_indeces], t2[train_indeces]
    t1_val, t2_val = t1[val_indeces], t2[val_indeces]

    return t1_train, t1_val, t2_train, t2_val


#Train the content AutoEncoder
## Determine whether to use vector or tensor reconstructor
content_trainer = train_vector_reconstructor if train_content.dim() == 2 else train_tensor_reconstructor

print('Start training content Autoencoder...')
content_reconstructor = content_trainer(train_content, train_images)
print('Content Autoencoder is trained!')

print('Start training content bias Autoencoder...')
content_bias_reconstructor = content_trainer(train_content_bias, train_images)
print('Content bias Autoencoder is trained!')

#Train the style Decoder
print('----------------------------------------')
style_trainer = train_vector_reconstructor if train_style.dim() == 2 else train_tensor_reconstructor

print('Start training style Autoencoder...')
style_reconstructor = style_trainer(train_style, train_images)
print('Style Autoencoder is trained!')

print('Start training style bias Autoencoder...')
style_bias_reconstructor = style_trainer(train_style_bias, train_images)
print('Style bias autoencoder is trained!')

# Test content Autoencoder
print('----------------------------------------')
content_tester = test_vector_reconstructor if test_content.dim() == 2 else test_tensor_reconstructor

print('Start testing content Autoencoder...')
content_loss = content_tester(content_reconstructor, test_content, test_images)
print(f'Content Autoencoder is tested! MSE: {content_loss}')

print('Start testing content bias Autoencoder...')
content_bias_loss = content_tester(content_bias_reconstructor, test_content_bias, test_images)
print(f'Content bias Autoencoder is tested! MSE: {content_bias_loss}')

# Test style autoencoder
print('----------------------------------------')
style_tester = test_vector_reconstructor if test_style.dim() == 2 else test_tensor_reconstructor

print('Start testing Style Autoencoder...')
style_loss = style_tester(style_reconstructor, test_style, test_images)
print(f'Style Autoencoder is tested! MSE: {style_loss}')

print('Start testing style bias Autoencoder...')
style_bias_loss = style_tester(style_bias_reconstructor, test_style_bias, test_images)
print(f'Style bias Autoencoder is tested! MSE: {style_bias_loss}')

# Store results
IoBc = content_bias_loss / content_loss
IoBs = style_bias_loss / style_loss

print(f'IoBc is {IoBc}, IoBs is {IoBs}')
file = open(result_file, 'a')
file.write('\nIoB metric for ' + dir_root + ':\n')
file.write('MSE Content Bias: ' + str(content_bias_loss) + '\n')
file.write('MSE Content: ' + str(content_loss) + '\n')
file.write('MSE Style Bias: ' + str(style_bias_loss) + '\n')
file.write('MSE Style: ' + str(style_loss) + '\n')
file.write('IoBc: ' + str(IoBc) + '\n')
file.write('IoBs: ' + str(IoBs) + '\n')
file.close()
