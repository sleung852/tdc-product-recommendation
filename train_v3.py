import pandas as pd
import numpy as np
from model.gmf import GMFEngine
from model.mlp import MLPEngine
from model.neumf import NeuMFEngine
from data import SampleGenerator
import os

import torch
torch.cuda.is_available()

import argparse

# procedures on training each model
def train_model(model, config):
    engine = model(config)
    best_hit = 0
    for epoch in range(config['num_epoch']):
        print('Epoch {} starts !'.format(epoch))
        print('-' * 70)
        train_loader = sample_generator.instance_a_train_loader(config['num_negative'], config['batch_size'])
        engine.train_an_epoch(train_loader, epoch_id=epoch)
        hit_ratio, ndcg = engine.evaluate(evaluate_data, epoch_id=epoch)
        if epoch % 20 == 0:
            engine.save(config['alias'], epoch, hit_ratio, ndcg)
        elif (epoch == config['num_epoch'] - 1):
            engine.save(config['alias'], epoch, hit_ratio, ndcg)
        if hit_ratio > best_hit:
            best_hit = hit_ratio
            engine.save(config['alias'], epoch, hit_ratio, ndcg, backup=False)
    print('Outputing the Best model')
    engine.full_save(config['alias'])
    return best_hit

#gmf configuration
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

# mlp configuration
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
              'pretrain_mf': 'gmf_factor8neg4-implict_best.model',
              'model_dir':'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'}

# neumf configuration
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
                'pretrain_mf': 'gmf_factor8neg4-implict_best.model',
                'pretrain_mlp': 'mlp_factor8neg4_bz256_166432168_pretrain_reg_0.0000001_best.model',
                'model_dir':'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'
                }

# train all three models - the entire training pipeline
def train_full_pipeline(data_path):
	# DataLoader for training
	sample_generator = SampleGenerator(data_path)
	## need to add asserts in SampleGenerator to check input format is correct
	evaluate_data = sample_generator.evaluate_data
	num_itemid, num_userid = sample_generator.usr_item_unique()
	gmf_best = train_model(GMFEngine, gmf_config)
	mlp_best = train_model(MLPEngine, mlp_config)
	neumf_best = train_model(NeuMFEngine, neumf_config)
	print('Done Training\n')
  print('** Result Report **\n')
  print('Stage 1 - GMF Hit Rate: {:.2f}%'.format(gmf_best))
  print('Stage 2 - MLP Hit Rate: {:.2f}%'.format(mlp_best))
  print('Stage 3 - NeuMF Hit Rate: {:.2f}%'.format(neumf_best))

# workflow
parser = argparse.ArgumentParser()
parser.add_argument("data_directory", help="state the directory of the csv data file", type=str)
args = parser.parse_args()

train_full_pipeline(args.data_directory)


