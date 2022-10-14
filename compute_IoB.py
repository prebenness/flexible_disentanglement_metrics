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

#Transpose data for pytorch
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

num_epochs = 40
batch_size = 10
num_itr_train = int(train_num_samples / batch_size)


def train_tensor_reconstructor(repr, target):
    reconstructor = TensorReconstructor()
    reconstructor.to(device)
    for epoch in range(num_epochs):
        index = torch.randperm(train_num_samples)
        for i in range(num_itr_train):
            repr_batch = repr[index[i * batch_size:(i+1)*batch_size], :, :, :].float().to(device)
            target_batch = target[index[i * batch_size:(i+1)*batch_size], :, :, :].float().to(device)
            mse_loss = reconstructor.AE_update(repr_batch, target_batch)
        print(f'Epoch {epoch}, MSE: {mse_loss}')
    
    return reconstructor


def train_vector_reconstructor(repr, target):
    reconstructor = VectorReconstructor(input_dim=repr.shape[-1], output_dim=target.shape[-1])
    reconstructor.to(device)

    summary(reconstructor, (repr.shape[-1],))

    for epoch in range(num_epochs):
        index = torch.randperm(train_num_samples)
        for i in range(num_itr_train):
            repr_batch = repr[index[i * batch_size:(i+1)*batch_size], :].float().to(device)
            target_batch = target[index[i * batch_size:(i+1)*batch_size], :, :, :].float().to(device)
            mse_loss = reconstructor.DE_update(repr_batch, target_batch)
        print(f'Epoch {epoch}, MSE: {mse_loss}')
    
    return reconstructor


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
        print(f'Test loss MSE: {mse_av}')

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
        print(f'Test loss MSE: {mse_av}')

    return mse_av


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
print('Content Autoencoder is tested!')

print('Start testing content bias Autoencoder...')
content_bias_loss = content_tester(content_bias_reconstructor, test_content_bias, test_images)
print('Content bias Autoencoder is tested!')

# Test style autoencoder
print('----------------------------------------')
style_tester = test_vector_reconstructor if test_style.dim() == 2 else test_tensor_reconstructor

print('Start testing Style Autoencoder...')
style_loss = style_tester(style_reconstructor, test_style, test_images)
print('Style Autoencoder is tested!')

print('Start testing style bias Autoencoder...')
style_bias_loss = style_tester(style_bias_reconstructor, test_style_bias, test_images)
print('Style bias Autoencoder is tested!')

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
