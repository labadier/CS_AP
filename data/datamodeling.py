#%%
from numpy.random import random
import pandas as pd, numpy as np, glob
import xml.etree.ElementTree as XT
from matplotlib import pyplot as plt
from collections import Counter
from sklearn.model_selection import StratifiedKFold
import os 

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

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
    
def read_truth(data_path):
    
    with open(data_path + '/truth.txt') as target_file:

        target = {}

        for line in target_file:
            inf = line.split(':::')
            target[inf[0]] = int(inf[1])

    return target

def load_Profiling_Data(data_path, labeled=True):

    addrs = np.array(glob.glob(data_path + '/*.xml'));addrs.sort()

    label = []
    tweets = []

    if labeled == True:
        target = read_truth(data_path)

    for adr in addrs:

        author = adr[len(data_path)+1: len(adr) - 4]
        tree = XT.parse(adr)
        root = tree.getroot()[0]
        siz = 0
        for twit in root:
            tweets.append(twit.text)
            siz +=1
        if labeled == True:
            label += [target[author]]*siz

    print(f'{bcolors.OKBLUE}Loaded {len(tweets)} tweets{bcolors.ENDC}' )
    return np.array(tweets), np.array(label)

def save_encoder_train_with_pdata(path):

    tweets, labels = load_Profiling_Data(path)
    m = np.random.permutation(len(tweets))
    tweets = tweets[m]
    labels = labels[m]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state = 23)

    for i, (train_index, test_index) in enumerate(skf.split(tweets, labels)):  
        dev = pd.DataFrame({'tweets':tweets[test_index], 'label':labels[test_index]})
        dev.to_csv(os.path.join(path,'pdata_dev.csv'))
        train = pd.DataFrame({'tweets':tweets[train_index], 'label':labels[train_index]})
        train.to_csv(os.path.join(path,'pdata_train.csv'))
        break
    
save_hate('tasks/hater/en/')
save_hate('tasks/hater/es/')
save_fake_en('tasks/faker/en/')
save_bot_en('tasks/bot/en/')
save_encoder_train_with_pdata('profiling/faker/train/en')


# %%
