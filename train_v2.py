import pandas as pd
import numpy as np
from gmf import GMFEngine
from mlp import MLPEngine
from neumf import NeuMFEngine
from data import SampleGenerator
import os

import torch
assert torch.cuda.is_available(), "GPU is not available"

# Load Data
data_dir = 'data/hktdc/cleaned_data.csv'
tdc_record = pd.read_csv(data_dir, names=['uid', 'mid', 'rating', 'timestamp'],  engine='python')

# Reindex
tdc_record = tdc_record.iloc[1:,:]
user_id = tdc_record[['uid']].drop_duplicates().reindex()
user_id['userId'] = np.arange(len(user_id))
tdc_record = pd.merge(tdc_record, user_id, on=['uid'], how='left')
item_id = tdc_record[['mid']].drop_duplicates()
item_id['itemId'] = np.arange(len(item_id))
tdc_record = pd.merge(tdc_record, item_id, on=['mid'], how='left')
tdc_record = tdc_record[['userId', 'itemId', 'rating', 'timestamp']]

tdc_record['rating']=tdc_record['rating'].astype('int32')
tdc_record['timestamp']=tdc_record['timestamp'].astype('float64')

print('Range of userId is [{}, {}]'.format(tdc_record.userId.min(), tdc_record.userId.max()))
print('Range of itemId is [{}, {}]'.format(tdc_record.itemId.min(), tdc_record.itemId.max()))
print(tdc_record.dtypes)

num_itemid=len(tdc_record['itemId'].unique())
num_userid=len(tdc_record['userId'].unique())

# DataLoader for training
sample_generator = SampleGenerator(ratings=tdc_record)
evaluate_data = sample_generator.evaluate_data

# Training Engine

def train_model(model, config):
    engine = model(config)
    for epoch in range(config['num_epoch']):
        print('Epoch {} starts !'.format(epoch))
        print('-' * 80)
        train_loader = sample_generator.instance_a_train_loader(config['num_negative'], config['batch_size'])
        engine.train_an_epoch(train_loader, epoch_id=epoch)
        hit_ratio, ndcg = engine.evaluate(evaluate_data, epoch_id=epoch)
        engine.save(config['alias'], epoch, hit_ratio, ndcg)

#setup configuration for GMF
gmf_config = {'alias': 'gmf_factor8neg4-implict',
              'num_epoch': 200,
              'batch_size': 4,
              # 'optimizer': 'sgd',
              # 'sgd_lr': 1e-3,
              # 'sgd_momentum': 0.9,
              # 'optimizer': 'rmsprop',
              # 'rmsprop_lr': 1e-3,
              # 'rmsprop_alpha': 0.99,
              # 'rmsprop_momentum': 0,
              'optimizer': 'adam',
              'adam_lr': 1e-3,
              'num_users': num_userid,
              'num_items': num_itemid,
              'latent_dim': 8,
              'num_negative': 4,
              'l2_regularization': 0, # 0.01
              'use_cuda': True,
              'device_id': 0,
              'model_dir':'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'}




# Train GMF Model
train_model(GMFEngine, gmf_config)

#find file name
gmf_model='tbc'
for file in os.listdir('checkpoints/'):
    leng= len('gmf_factor8neg4-implict_Epoch199')
    if file[:leng]=='gmf_factor8neg4-implict_Epoch199':
        print (file)
        gmf_model=file
        break

mlp_config = {'alias': 'mlp_factor8neg4_bz256_166432168_pretrain_reg_0.0000001',
              'num_epoch': 200,
              'batch_size': 4,  # 1024,
              'optimizer': 'adam',
              'adam_lr': 1e-3,
              'num_users': num_userid,
              'num_items': num_itemid,
              'latent_dim': 8,
              'num_negative': 4,
              'layers': [16,64,32,16,8],  # layers[0] is the concat of latent user vector & latent item vector
              'l2_regularization': 0.0000001,  # MLP model is sensitive to hyper params
              'use_cuda': True,
              'device_id': 0,
              'pretrain': True,
              'pretrain_mf': os.path.join('checkpoints',gmf_model),
              'model_dir':'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'}

train_model(MLPEngine, mlp_config)
#need to add a learning rate scheduler

mlp_model='tbc'
for file in os.listdir('checkpoints/'):
    leng= len('mlp_factor8neg4_bz256_166432168_pretrain_reg_0.0000001_Epoch199')
    if file[:leng]=='mlp_factor8neg4_bz256_166432168_pretrain_reg_0.0000001_Epoch162':
        print (file)
        mlp_model=file
        break

neumf_config = {'alias': 'pretrain_neumf_factor8neg4',
                'num_epoch': 200,
                'batch_size': 4,
                'optimizer': 'adam',
                'adam_lr': 1e-3,
                'num_users': num_userid,
              'num_items': num_itemid,
                'latent_dim_mf': 8,
                'latent_dim_mlp': 8,
                'num_negative': 4,
                'layers': [16,64,32,16,8],  # layers[0] is the concat of latent user vector & latent item vector
                'l2_regularization': 0.0000001,
                'use_cuda': True,
                'device_id': 0,
                'pretrain': True,
                'pretrain_mf': 'checkpoints/{}'.format(gmf_model),
                'pretrain_mlp': 'checkpoints/{}'.format(mlp_model),
                'model_dir':'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'
                }

train_model(NeuMFEngine, neumf_config)

print('Done!')