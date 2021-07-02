#%%
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from collections import Counter

def partition(data, split, criterion):
    times = Counter(data[:,criterion])
    for i in times:
        times[i] = int(times[i]*split)
    
    train = []
    dev = []
    for i in data:
        if times[i[criterion]]:
            train.append(i[0])
            times[i[criterion]] -= 1
        else: dev.append(i[0])
    return train, dev

def save_csvs(pclass, nclass, split, path, criterion):

    faketrain, fakedev = partition(pclass, split, criterion)
    nofaketrain, nofakedev = partition(nclass, split, criterion)

    train_labels = np.concatenate([np.ones((len(faketrain),), dtype=int), np.zeros((len(nofaketrain),), dtype=int)]) 
    dev_labels = np.concatenate([np.ones((len(fakedev),), dtype=int), np.zeros((len(nofakedev),), dtype=int)]) 
    train_examples = np.array(faketrain+nofaketrain)
    dev_examples = np.array(fakedev+nofakedev)
    
    perm = np.random.permutation(len(train_examples))
    train = pd.DataFrame({'tweets':train_examples[perm], 'label':train_labels[perm]})
    train.to_csv(path+'train.csv')
    perm = np.random.permutation(len(dev_labels))
    dev = pd.DataFrame({'tweets':dev_examples[perm], 'label':dev_labels[perm]})
    dev.to_csv(path+'dev.csv')

def save_fake_en(path):
    iter_csv = pd.read_csv(path + 'claimskg_result.csv', iterator=True, chunksize=1000, usecols=['text', 'ratingName', 'source'])
    fake = pd.concat([chunk[chunk['ratingName'] == 'FALSE'] for chunk in iter_csv]).to_numpy()
    iter_csv = pd.read_csv(path + 'claimskg_result.csv', iterator=True, chunksize=1000, usecols=['text', 'ratingName', 'source'])
    nofake = pd.concat([chunk[chunk['ratingName'] == 'TRUE'] for chunk in iter_csv]).to_numpy()
    save_csvs(fake, nofake, 0.8, path, 2)

def save_bot_en(path):
    iter_csv = pd.read_csv(path + 'training_data_2_csv_UTF.csv', iterator=True, chunksize=1000, usecols=['description', 'bot', 'verified'])
    bot = pd.concat([chunk[chunk['bot'] == 1] for chunk in iter_csv]).to_numpy()
    iter_csv = pd.read_csv(path + 'training_data_2_csv_UTF.csv', iterator=True, chunksize=1000, usecols=['description', 'bot', 'verified'])
    nobot = pd.concat([chunk[chunk['bot'] == 0] for chunk in iter_csv]).to_numpy()
    save_csvs(bot, nobot, 0.8, path, 1)

def save_hate(path):
    dev = pd.read_csv(path + f'hateval2019_{path[-3:-1]}_dev.csv', usecols=['text', 'HS']).to_numpy()
    train = pd.read_csv(path + f'hateval2019_{path[-3:-1]}_train.csv', usecols=['text', 'HS']).to_numpy()
    dev = pd.DataFrame({'tweets':dev[:,0], 'label':dev[:,1]})
    dev.to_csv(path+'dev.csv')
    train = pd.DataFrame({'tweets':train[:,0], 'label':train[:,1]})
    train.to_csv(path+'train.csv')
    
save_hate('tasks/hater/en/')
save_hate('tasks/hater/es/')
save_fake_en('tasks/faker/en/')
save_bot_en('tasks/bot/en/')
# %%
