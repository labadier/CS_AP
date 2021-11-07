#%%
from sklearn import model_selection
import torch, os, sys
sys.path.append('../')
from models.models import seed_worker
import numpy as np, pandas as pd
from models.classifiers import AttentionLSTM
from torch.utils.data import Dataset, DataLoader, dataloader
from sklearn.model_selection import StratifiedKFold
from utils import bcolors


class CW_Data(Dataset):

  def __init__(self, data):

    self.wordl = data[0] 
    self.charl = data[1] 
    self.label = data[2]

  def __len__(self):
    return self.wordl.shape[0]

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()

    tweetword = self.wordl[idx] 
    tweetchar = self.charl[idx] 
    label = self.label[idx]

    sample = {'word': tweetword, 'char':tweetchar, 'label':label}
    return sample

class CNN_LSTM(torch.nn.Module):

  def __init__(self, embedding_matrix, fix_emb, lstm_layer=64):

    super(CNN_LSTM, self).__init__()
    self.emb = torch.nn.Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1])
    self.emb.load_state_dict({'weight': torch.from_numpy(embedding_matrix)})
    if fix_emb:
      self.emb.weight.requires_grad = False
    else: self.emb.weight.requires_grad = True

    self.conv1 = torch.nn.Sequential(torch.nn.Conv1d(in_channels=embedding_matrix.shape[1], out_channels=32, kernel_size=5),
                                      torch.nn.Dropout(p = 0.4),
                                      torch.nn.ReLU(),
                                      torch.nn.MaxPool1d(5))
    self.conv2 = torch.nn.Sequential(torch.nn.Conv1d(in_channels=embedding_matrix.shape[1], out_channels=32, kernel_size=4),
                                      torch.nn.Dropout(p = 0.4),
                                      torch.nn.ReLU(),
                                      torch.nn.MaxPool1d(4))
    self.conv3 = torch.nn.Sequential(torch.nn.Conv1d(in_channels=embedding_matrix.shape[1], out_channels=32, kernel_size=3),
                                      torch.nn.Dropout(p = 0.4),
                                      torch.nn.ReLU(),
                                      torch.nn.MaxPool1d(3))

    self.att = AttentionLSTM(neurons=32, dimension=32)

    self.lstm = torch.nn.LSTM(batch_first=True, input_size=32, hidden_size=lstm_layer, bidirectional=False, proj_size=0)
    self.linear = torch.nn.Sequential(torch.nn.Dropout(p = 0.4),
                                      torch.nn.Linear(lstm_layer, 64),
                                      torch.nn.LeakyReLU())

  def forward(self, X):
    X = self.emb(X)
    X = torch.swapaxes(X, -1, -2)
    
    X1 = self.conv1(X)
    X2 = self.conv2(X)
    X3 = self.conv3(X)
    X = torch.cat([X1, X2, X3], dim = -1)
    X = torch.swapaxes(X, -1, -2)

    X =  self.att(X)
    X,_ = self.lstm(X)
    
    return self.linear(X[:,-1])

class SeqEncoder(torch.nn.Module):

  def __init__(self, language, embedding_matrix_word, lstm_layer=64):

    super(SeqEncoder, self).__init__()
    self.lang = language
    self.best_acc = None
    self.best_acc_train = None
    self.Word_EncMod = CNN_LSTM(embedding_matrix_word, True, 64)
    self.Char_EncMod = CNN_LSTM(np.random.randn(29, 100)*0.01,False, 64)

    self.dense = torch.nn.Sequential(torch.nn.Linear(lstm_layer*2, 64), 
                                      torch.nn.LeakyReLU())
    self.classifier = torch.nn.Linear(64, 2)  
    
    self.loss_criterion = torch.nn.CrossEntropyLoss() 
    self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    self.to(device=self.device)

  def forward(self, X_W, X_C, training = True):

    X_W = self.Word_EncMod(X_W.to(self.device))
    X_C = self.Char_EncMod(X_C.to(self.device))
    
    X = torch.cat([X_W, X_C], dim = -1)
    X = self.dense(X)

    if training == False:
      return X
    return self.classifier(X)

  def load(self, path):
    self.load_state_dict(torch.load(path, map_location=self.device))

  def save(self, path):
    if os.path.exists('./logs') == False:
        os.system('mkdir logs')
    torch.save(self.state_dict(), os.path.join('logs', path))


  def get_encodings(self, data, batch_size):

    self.eval()    
    data.append(np.zeros((len(data[0]),)))
    devloader = DataLoader(CW_Data(data), batch_size=batch_size, shuffle=False, num_workers=4)
 
    with torch.no_grad():
      out = None
      log = None
      for k, data in enumerate(devloader, 0):
        torch.cuda.empty_cache() 
        inputsw, inputsc = data['word'], data['char']

        dev_encode = self.forward(inputsw, inputsc, False)
        if k == 0:
          out = dev_encode
        else: 
          out = torch.cat((out, dev_encode), 0)

    out = out.cpu().numpy()
    del devloader
    return out

def train_Seq(model, data, language, model_name, splits = 5, epoches = 4, batch_size = 64, lr = 1e-3,  decay=1e-5):
 
  skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state = 23)
  history = []
  overall_acc = 0
  last_printed = None

  spl = 0
  for i, (train_index, test_index) in enumerate(skf.split(data[0], data[-1])):  
    
    history.append({'loss': [], 'acc':[], 'dev_loss': [], 'dev_acc': []})
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)
    trainloader = DataLoader(CW_Data([data[0][train_index], data[1][train_index], data[2][train_index]]), batch_size=batch_size, shuffle=True, num_workers=4, worker_init_fn=seed_worker)
    devloader = DataLoader(CW_Data([data[0][test_index], data[1][test_index], data[2][test_index]]), batch_size=batch_size, shuffle=True, num_workers=4, worker_init_fn=seed_worker)
    batches = len(trainloader)

    for epoch in range(epoches):

      running_loss = 0.0
      perc = 0
      acc = 0

      model.train()
      
      for j, data in enumerate(trainloader, 0):

        torch.cuda.empty_cache()         
        inputsw, inputsc, labels = data['word'], data['char'], data['label'].to(model.device)      
        
        optimizer.zero_grad()
        outputs = model(inputsw, inputsc)
        loss = model.loss_criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        # print statistics
        with torch.no_grad():
          if j == 0:
            acc = ((1.0*(torch.max(outputs, 1).indices == labels)).sum()/len(labels)).cpu().numpy()
            running_loss = loss.item()
          else: 
            acc = (acc + ((1.0*(torch.max(outputs, 1).indices == labels)).sum()/len(labels)).cpu().numpy())/2.0
            running_loss = (running_loss + loss.item())/2.0

        if (j+1)*100.0/batches - perc  >= 1 or j == batches-1:
          perc = (1+j)*100.0/batches
          last_printed = f'\rEpoch:{epoch+1:3d} of {epoches} step {j+1} of {batches}. {perc:.1f}% loss: {running_loss:.3f}'
          print(last_printed, end="")

      model.eval()
      history[-1]['loss'].append(running_loss)
      with torch.no_grad():
        out = None
        log = None
        for k, data in enumerate(devloader, 0):
          torch.cuda.empty_cache() 
          inputsw, inputsc, labels = data['word'], data['char'], data['label'].to(model.device)      

          dev_out = model(inputsw, inputsc)
          if k == 0:
            out = dev_out
            log = labels
          else: 
            out = torch.cat((out, dev_out), 0)
            log = torch.cat((log, labels), 0)

        dev_loss = model.loss_criterion(out, log).item()
        dev_acc = ((1.0*(torch.max(out, 1).indices == log)).sum()/len(log)).cpu().numpy()
        history[-1]['acc'].append(acc)
        history[-1]['dev_loss'].append(dev_loss)
        history[-1]['dev_acc'].append(dev_acc) 

        band = False
        if model.best_acc is None or model.best_acc < dev_acc:
          model.save(f'{model_name}_{language}_{i+1}.pt')
          model.best_acc = dev_acc
          model.best_acc_train = acc
          band = True
        elif model.best_acc == dev_acc and (model.best_acc_train is None or model.best_acc_train < acc):
          model.save(f'{model_name}_{language}_{i+1}.pt')
          model.best_acc_train = acc
          band = True

        ep_finish_print = f' acc: {acc:.3f} | dev_loss: {dev_loss:.3f} dev_acc: {dev_acc.reshape(-1)[0]:.3f}'

        if band == True:
            print(bcolors.OKBLUE + bcolors.BOLD + last_printed + ep_finish_print + '\t[Weights Updated]' + bcolors.ENDC)
        else: print(ep_finish_print)
                  
    overall_acc += model.best_acc
    print('Training Finished Split: {}'. format(i+1))
    del trainloader
    del model
    del devloader
    spl += 1
    break
  print(f"{bcolors.OKGREEN}{bcolors.BOLD}{50*'*'}\nOveral Accuracy {model_name} {language} in {spl} slpits: {overall_acc/spl}\n{50*'*'}{bcolors.ENDC}")
  return history









