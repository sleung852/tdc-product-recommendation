import torch
import random
import pandas as pd
from copy import deepcopy
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os
import numpy as np

tqdm.pandas()

random.seed(0)

class UserItemRatingDataset(Dataset):
    """Wrapper, convert <user, item, rating> Tensor into Pytorch Dataset"""
    def __init__(self, user_tensor, item_tensor, target_tensor):
        """
        args:

            target_tensor: torch.Tensor, the corresponding rating for <user, item> pair
        """
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.user_tensor[index], self.item_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.user_tensor.size(0)


class SampleGenerator:
    """Construct dataset for NCF"""

    def __init__(self, raw_data_dir):
        """
        args:
            ratings: pd.DataFrame, which contains 4 columns = ['userId', 'itemId', 'rating', 'timestamp']
        """
        print('\nInitialising data cleaning')
        self.ratings = self._clean_raw_data(pd.read_csv(raw_data_dir))
        assert 'userId' in self.ratings.columns
        assert 'itemId' in self.ratings.columns
        assert 'rating' in self.ratings.columns
        print('\nInitialising data pre-processing')
        # explicit feedback using _normalize and implicit using _binarize
        # self.preprocess_ratings = self._normalize(ratings)
        print('begin preprocess_ratings')
        self.preprocess_ratings = self._binarize(self.ratings)
        print('begin setting pools')
        self.user_pool = set(self.ratings['userId'].unique())
        self.item_pool = set(self.ratings['itemId'].unique())
        # create negative item samples for NCF learning
        print('creating negative items')
        self.negatives = self._sample_negative(self.ratings)
        print('split_loo')
        self.train_ratings, self.test_ratings = self._split_loo(self.preprocess_ratings)
        print('Done!')
        
        
    def _clean_raw_data(self, tdc_record):
        #tdc_record=pd.read_csv('user-item-oct-dec-2018-v2-modified.csv')

        tdc_record.drop(tdc_record[tdc_record[['USER_ID','ITEM_ID']].duplicated()].index, inplace=True)

        def check_users_items_1(df):
            # user side
            user_id = df.groupby('USER_ID').count()['ITEM_ID']
            bad_users=user_id[user_id==1].index
            # item side
            item_id = df.groupby('ITEM_ID').count()['USER_ID']
            bad_items =item_id[item_id==1].index
            return bad_users, bad_items

        def clear_users_items_1(df, bad_users, bad_items):
            # iterrows
            drop_index=[]
            for ind, row in df.iterrows():
                if (row['USER_ID'] in bad_users) | (row['ITEM_ID'] in bad_items):
                    drop_index.append(ind)
            df_cleaned = df.drop(drop_index)
            return df_cleaned

        bad_users, bad_items = check_users_items_1(tdc_record)
        count = 1
        while ((len(bad_users)>0) | (len(bad_items)>0)):
            print('Number of bad users: {} \nNumber of bad items: {}'.format(len(bad_users), len(bad_items)))
            print('Starting clean process {}\n'.format(count))
            tdc_record = clear_users_items_1(tdc_record, bad_users, bad_items)
            tdc_record.drop_duplicates(inplace=True)
            bad_users, bad_items = check_users_items_1(tdc_record)
            count += 1
        print('No more rubbish data.')
        tdc_record.to_csv(os.path.join('data','cleaned_data.csv'), index=False)
        tdc_record.columns=['uid', 'mid', 'timestamp']

        # Reindex
        tdc_record = tdc_record.iloc[1:,:]
        user_id = tdc_record[['uid']].drop_duplicates().reindex()
        user_id['userId'] = np.arange(len(user_id))
        tdc_record = pd.merge(tdc_record, user_id, on=['uid'], how='left')
        item_id = tdc_record[['mid']].drop_duplicates()
        item_id['itemId'] = np.arange(len(item_id))
        tdc_record = pd.merge(tdc_record, item_id, on=['mid'], how='left')
        tdc_record['rating']=1.0
        tdc_record = tdc_record[['userId', 'itemId', 'rating', 'timestamp']]
        
        tdc_record.drop_duplicates(inplace=True)

        tdc_record['rating']=tdc_record['rating'].astype('int32')
        tdc_record['timestamp']=tdc_record['timestamp'].astype('float64')

        print('Range of userId is [{}, {}]'.format(tdc_record.userId.min(), tdc_record.userId.max()))
        print('Range of itemId is [{}, {}]'.format(tdc_record.itemId.min(), tdc_record.itemId.max()))
        print(tdc_record.dtypes)
        
        return tdc_record
    
    def usr_item_unique(self):
        num_itemid=len(self.ratings['itemId'].unique())
        num_userid=len(self.ratings['userId'].unique()) 
        return num_itemid, num_userid

    def _normalize(self, ratings):
        """
        func: normalize into [0, 1] from [0, max_rating], explicit feedback
        notes: in the future when the interaction datas are more complicated,
        you can adjust the interactions into two types. one is view, one is message
        sent to supplier and one is product purchased
        """
        ratings = deepcopy(ratings)
        max_rating = ratings.rating.max()
        ratings['rating'] = ratings.rating * 1.0 / max_rating
        return ratings
    
    def _binarize(self, ratings):
        """binarize into 0 or 1, imlicit feedback"""
        ratings = deepcopy(ratings)
        #ratings['rating'][ratings['rating'] > 0] = 1.0
        _rate = (ratings['rating'] > 0).values * 1.0
        ratings['rating'] = _rate
        return ratings

    def _split_loo(self, ratings):
        """leave one out train/test split """
        ratings['rank_latest'] = ratings.groupby(['userId'])['timestamp'].rank(method='first', ascending=False)
        test = ratings[ratings['rank_latest'] == 1]
        train = ratings[ratings['rank_latest'] > 1]
        #test what if i took this away
        assert train['userId'].nunique() == test['userId'].nunique()
        return train[['userId', 'itemId', 'rating']], test[['userId', 'itemId', 'rating']]

    def _sample_negative(self, ratings):
        """return all negative items & 100 sampled negative items"""
        print('0')
        interact_status = ratings.groupby('userId')['itemId'].apply(set).reset_index().rename(
            columns={'itemId': 'interacted_items'})
        print('1')
        # find out which products have not been interacted by each of the users
        interact_status['negative_items'] = interact_status['interacted_items'].progress_apply(lambda x: self.item_pool.difference(x))
        print('2')
        interact_status['negative_samples'] = interact_status['negative_items'].progress_apply(lambda x: random.sample(x, 99))
        return interact_status[['userId', 'negative_items', 'negative_samples']]

    def instance_a_train_loader(self, num_negatives, batch_size):
        """instance train loader for one training epoch"""
        users, items, ratings = [], [], []
        train_ratings = pd.merge(self.train_ratings, self.negatives[['userId', 'negative_items']], on='userId')
        train_ratings['negatives'] = train_ratings['negative_items'].apply(lambda x: random.sample(x, num_negatives))
        for row in train_ratings.itertuples():
            users.append(int(row.userId))
            items.append(int(row.itemId))
            ratings.append(float(row.rating))
            for i in range(num_negatives):
                users.append(int(row.userId))
                items.append(int(row.negatives[i]))
                ratings.append(float(0))  # negative samples get 0 rating
        dataset = UserItemRatingDataset(user_tensor=torch.LongTensor(users),
                                        item_tensor=torch.LongTensor(items),
                                        target_tensor=torch.FloatTensor(ratings))
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    @property
    def evaluate_data(self):
        """create evaluate data"""
        test_ratings = pd.merge(self.test_ratings, self.negatives[['userId', 'negative_samples']], on='userId')
        test_users, test_items, negative_users, negative_items = [], [], [], []
        print('Begin the loop...')
        for row in test_ratings.itertuples():
            test_users.append(int(row.userId))
            test_items.append(int(row.itemId))
            for i in range(len(row.negative_samples)):
                negative_users.append(int(row.userId))
                negative_items.append(int(row.negative_samples[i]))
        return [torch.LongTensor(test_users), torch.LongTensor(test_items), torch.LongTensor(negative_users),
                torch.LongTensor(negative_items)]
