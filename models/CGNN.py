import os, sys, numpy as np, torch
from torch.functional import split
sys.path.append('../')
import torch_geometric
from sklearn.model_selection import StratifiedKFold
from utils import bcolors
from models.models import seed_worker

class GCN(torch.nn.Module):

  def __init__(self, language, hidden_channels=64, features_nodes=96, handed_features = False):
    super(GCN, self).__init__()

    self.conv1 = torch_geometric.nn.GCNConv(features_nodes, hidden_channels)
    self.conv2 = torch_geometric.nn.GCNConv(hidden_channels, hidden_channels)
    # self.conv2 = torch_geometric.nn.ChebConv(hidden_channels, hidden_channels, 3)#3torch_geometric.nn.GCNConv(hidden_channels, hidden_channels)
    # self.conv3 = torch_geometric.nn.ChebConv(hidden_channels, hidden_channels, 2)#2
    self.lin = torch.nn.Linear(hidden_channels + 32*handed_features, 64)
    self.pred = torch.nn.Sequential(torch.nn.LeakyReLU(),  torch.nn.Linear(64, 32), torch.nn.Linear(32, 2))
    self.best_acc = None
    self.best_acc_train = None
    self.language = language
    self.loss_criterion = torch.nn.CrossEntropyLoss()
    self.dense_features = torch.nn.Linear(in_features=177, out_features=32)
    self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    self.to(device=self.device)

  def forward(self, x, edge_index, batch, phase='train', F = None ):

    edge_index = edge_index.to(self.device)
    x = self.conv1(x.to(self.device), edge_index)
    x = x.relu()
    x = self.conv2(x, edge_index)
    x = x.relu()
    # x = self.conv3(x, edge_index)
    x = torch_geometric.nn.global_mean_pool(x, batch.to(self.device)) 

    x = torch.nn.functional.dropout(x, p=0.2, training=self.training)

    if F is not None:
      # print(x.shape, F.shape)
      F = F.reshape((x.shape[0], -1))
      F = self.dense_features(F.to(device=self.device))
      x = torch.cat([x, F], dim = -1)

    x = self.lin(x)
    if phase == 'encode':
      return x
    return self.pred(x)
		

  def load(self, path):
    self.load_state_dict(torch.load(path, map_location=self.device))

  def save(self, path):
    torch.save(self.state_dict(), path)

  def get_encodings(self, encodings, rep):

    self.eval() 
    if 'h' in rep:
      features = torch.tensor(encodings[1], dtype=torch.float)   
    else: features = torch.zeros((len(encodings[0]), ))
    
    a = []
    b = []
    for i in range(encodings[0].shape[1]):  
      a += [i]*int(encodings[0].shape[1])
      b += [j for j in range(encodings[0].shape[1])]

    edges = [a, b]
    edges = torch.tensor(edges, dtype=torch.long)
    encodings = torch.tensor(encodings[0], dtype=torch.float)
    
    data_test = [torch_geometric.data.Data(x=encodings[i], y = features[i], edge_index=edges) for i in range(encodings.shape[0])]
    devloader = torch_geometric.data.DataLoader(data_test, batch_size=64, shuffle=False, num_workers=4, worker_init_fn=seed_worker)
		

    with torch.no_grad():
      out = None
      for k, data in enumerate(devloader, 0):
        torch.cuda.empty_cache() 
        dev_out = self.forward(data.x, data.edge_index, data.batch, phase='encode', F = (data.y if 'h' in rep else None))
        if k == 0:
          out = dev_out
        else: 
          out = torch.cat((out, dev_out), 0)

    out = out.cpu().numpy()
    del devloader
    return out

def train_GCNN(rep, task, data_train, data_test, language, splits = 5, epoches = 4, batch_size = 64, hidden_channels = 64, lr = 1e-5,  decay=2e-5):
	
  a = []
  b = []
  for i in range(data_train[0].shape[1]):  
    a += [i]*int(data_train[0].shape[1])
    b += [j for j in range(data_train[0].shape[1])]

  edges_train = [a, b]
  edges_train = torch.tensor(edges_train, dtype=torch.long)
  target_train = torch.tensor(data_train[2])
  encodings_train = torch.tensor(data_train[0], dtype=torch.float)

  features_train = torch.tensor(data_train[1], dtype=torch.float)
  features_test = torch.tensor(data_test[1], dtype=torch.float)

  a = []
  b = []
  for i in range(data_test[0].shape[1]):  
    a += [i]*int(data_test[0].shape[1])
    b += [j for j in range(data_test[0].shape[1])]

  edges_test = [a, b]
  edges_test = torch.tensor(edges_test, dtype=torch.long)
  target_test = torch.tensor(data_test[2])
  encodings_test = torch.tensor(data_test[0], dtype=torch.float)


  history = [{'loss': [], 'acc':[], 'dev_loss': [], 'dev_acc': []}]
  model = GCN(language, hidden_channels, encodings_train.shape[-1], handed_features = ('h' in rep))

  optimizer = torch.optim.Adam(model.parameters(), lr=lr)
  
  data_train = [torch_geometric.data.Data(x=encodings_train[i], y=[target_train[i], features_train[i]], edge_index=edges_train) for i in range(len(encodings_train))]
  data_test = [torch_geometric.data.Data(x=encodings_test[i], y=[target_test[i], features_test[i]], edge_index=edges_test) for i in range(len(encodings_test))]
  
  trainloader = torch_geometric.data.DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=4, worker_init_fn=seed_worker)
  devloader = torch_geometric.data.DataLoader(data_test, batch_size=batch_size, shuffle=True, num_workers=4, worker_init_fn=seed_worker)
  batches = len(trainloader)

  for epoch in range(epoches):

    running_loss = 0.0
    perc = 0
    acc = 0
    
    model.train()
    last_printed = ''
    for j, data in enumerate(trainloader, 0):

      torch.cuda.empty_cache()         

      optimizer.zero_grad()
      outputs = model(data.x, data.edge_index, data.batch, F = (data.y[1] if 'h' in rep else None))
      loss = model.loss_criterion(outputs, data.y[0].to(model.device))
      
      loss.backward()
      optimizer.step()

      # print statistics
      outputs = outputs.argmax(dim=1).cpu()
      with torch.no_grad():
        if j == 0:
          acc = ((1.0*(outputs == data.y[0])).sum()/len(data.y[0])).numpy()
          running_loss = loss.item()
        else: 
          acc = (acc + ((1.0*(outputs == data.y[0])).sum()/len(data.y[0])).numpy())/2.0
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

        dev_out = model(data.x, data.edge_index, data.batch, F = (data.y[1] if 'h' in rep else None))
        if k == 0:
          out = dev_out
          log = data.y[0]
        else: 
          out = torch.cat((out, dev_out), 0)
          log = torch.cat((log, data.y[0]), 0)

      dev_loss = model.loss_criterion(out, log.to(model.device)).item()
      out = out.argmax(dim=1).cpu()
      dev_acc = ((1.0*(out == log)).sum()/len(log)).cpu().numpy() 
      history[-1]['acc'].append(acc)
      history[-1]['dev_loss'].append(dev_loss)
      history[-1]['dev_acc'].append(dev_acc) 

    band = False
    if model.best_acc is None or model.best_acc < dev_acc:
      model.save(f'logs/gcn_{language}_{rep}.pt')
      model.best_acc = dev_acc
      model.best_acc_train = acc
      band = True
    elif model.best_acc == dev_acc and (model.best_acc_train is None or model.best_acc_train < acc):
      model.save(f'logs/gcn_{language}_{rep}.pt')
      model.best_acc_train = acc
      band = True


    ep_finish_print = f' acc: {acc:.3f} | dev_loss: {dev_loss:.3f} dev_acc: {dev_acc.reshape(-1)[0]:.3f}'

    if band == True:
      print(bcolors.OKBLUE + bcolors.BOLD + last_printed + ep_finish_print + '\t[Weights Updated]' + bcolors.ENDC)
    else: print(ep_finish_print)

  print(f"{bcolors.OKGREEN}{bcolors.BOLD}{50*'*'}\n TASK: {task.upper()} MODEL: GCN REPRESENTATION: {rep} LRATE: {lr} ~~ LANGUGE: {language} BATCH: {batch_size}: {model.best_acc}\n{50*'*'}{bcolors.ENDC}")
  del trainloader
  del model
  del devloader
  return history


def predicgcn(encodings, idx, language, splits, output, batch_size, hidden_channels, save_predictions):

	a = []
	b = []
	for i in range(encodings.shape[1]):  
		a += [i]*int(encodings.shape[1])
		b += [j for j in range(encodings.shape[1])]

	edges = [a, b]
	edges = torch.tensor(edges, dtype=torch.long)
	encodings = torch.tensor(encodings, dtype=torch.float)

	model = GCN(language, hidden_channels, encodings.shape[-1])
	data_test = [torch_geometric.data.Data(x=encodings[i], edge_index=edges) for i in range(len(encodings))]
	devloader = torch_geometric.data.DataLoader(data_test, batch_size=batch_size, shuffle=True, num_workers=4, worker_init_fn=seed_worker)
	
	model.eval()
	y_hat = np.zeros((len(idx), ))
	for i in range(splits):

		model.load(f'logs/gcn_{language}_{i+1}.pt')
		with torch.no_grad():
			out = None
			ids = None
			for k, data in enumerate(devloader, 0): 
				dev_out = model(data.x, data.edge_index, data.batch)
				if k == 0:
					out = dev_out
				else:  out = torch.cat((out, dev_out), 0)

			y_hat += torch.argmax(torch.nn.functional.softmax(out, dim=-1), axis=-1).cpu().numpy()
	y_hat = np.int32(np.round(y_hat/splits, decimals=0))
	save_predictions(idx, y_hat, language, output)


