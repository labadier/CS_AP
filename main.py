#%%
import argparse, sys, os, numpy as np, torch, random
from matplotlib.pyplot import axis
from models.models import Encoder, train_Encoder
from utils import plot_training, load_Profiling_Data, make_pairs
from utils import make_triplets,make_profile_pairs, save_predictions, read_embedding, translate_char
from utils import make_pairs_with_protos, compute_centers_PSC, read_data, translate_words
from sklearn.metrics import f1_score
from models.classifiers import K_Impostor, train_classifier, predict, FNN_Classifier, LSTMAtt_Classifier
from models.classifiers import svm
from models.Sequential import SeqEncoder, train_Seq
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score
from utils import bcolors
import paramfile

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

def check_params(args=None):
  parser = argparse.ArgumentParser(description='Language Model Encoder')

  parser.add_argument('-l', metavar='language', default=paramfile.l, help='Task Language')
  parser.add_argument('-task', metavar='task', default=paramfile.l, help='Task')  
  parser.add_argument('-phase', metavar='phase', default=paramfile.phase, help='Phase')
  parser.add_argument('-rep', metavar='rep', help='Represenations to use')
  parser.add_argument('-output', metavar='output', help='Output Path')
  parser.add_argument('-lr', metavar='lrate', default = 1e-5, type=float, help='learning rate')
  parser.add_argument('-tmode', metavar='tmode', default = paramfile.tmode, help='Encoder Weights Mode')
  parser.add_argument('-decay', metavar='decay', default = 2e-5, type=float, help='learning rate decay')
  parser.add_argument('-splits', metavar='splits', default = 5, type=int, help='spits cross validation')
  parser.add_argument('-ml', metavar='max_length', default = paramfile.ml, type=int, help='Maximun Tweets Length')
  parser.add_argument('-interm_layer', metavar='int_layer', default = paramfile.interm_layer, type=int, help='Intermediate layers neurons')
  parser.add_argument('-epoches', metavar='epoches', default=12, type=int, help='Trainning Epoches')
  parser.add_argument('-bs', metavar='batch_size', default=paramfile.bs, type=int, help='Batch Size')
  parser.add_argument('-dp', metavar='data_path', default=paramfile.dp, help='Data Path')
  parser.add_argument('-mode', metavar='mode', default=paramfile.mode, help='Encoder Mode')#, choices=['tEncoder', 'tSiamese', 'eSiamese', 'encode', 'pEncoder', 'tPredictor', learnmetric])
  parser.add_argument('-wp', metavar='wp', help='Weight Path', default='logs' )
  parser.add_argument('-loss', metavar='loss', help='Loss for Siamese Architecture', default='contrastive', choices=['triplet', 'contrastive'] )
  parser.add_argument('-rp', metavar='randpro', help='Between 0 and 1 float to choose random prototype among examples', type=float, default=0.25)
  parser.add_argument('-metric', metavar='mtricImp', help='Metric to compare on Impostor Method', default='cosine', choices=['cosine', 'euclidean', 'deepmetric'] )
  parser.add_argument('-ecnImp', metavar='EncodertoImp', help='Encoder to use on Importor either Siamese or Transformer', default='transformer', choices=['transformer', 'siamese', 'fcnn', 'lstm', 'gcn'] )
  parser.add_argument('-dt', metavar='data_test', help='Get Data for test')
  parser.add_argument('-up', metavar='useof_prototype', help='Using Prototipes for Impostor or compare to random examples', default="prototipical", choices=["prototipical", "random"])
  parser.add_argument('-lstm_size', metavar='LSTM_hidden_size', type=int,help='LSTM classfifier hidden size')
  return parser.parse_args(args)


def get_encodings(model, mod_name):

  encodings_train = None
  encodings_dev = None

  handed_train = None
  handed_dev = None
  phanded_train = None
  phanded_dev = None

  if 't' in rep:
    encodings_train = torch.load(f'logs/transformers/{task}_train_encodings_{language[:2]}.pt')
    encodings_dev = torch.load(f'logs/transformers/{task}_dev_encodings_{language[:2]}.pt')
  elif 'c' in rep:
    encodings_train = torch.load(f'logs/cnn_lstm/{task}_train_encodings_{language[:2]}.pt')
    encodings_dev = torch.load(f'logs/cnn_lstm/{task}_dev_encodings_{language[:2]}.pt')

  if 'h' in rep:
    phanded_train = f'logs/handcrafted/{task}_train_{language[:2].lower()}.json'
    phanded_dev = f'logs/handcrafted/{task}_dev_{language[:2].lower()}.json'
     
  _, _, labels_train, handed_train = load_Profiling_Data(f'{data_path}/train/{language.lower()}', labeled=True, w_features = phanded_train )
  _, _, labels_dev, handed_dev = load_Profiling_Data(f'{data_path}/dev/{language.lower()}', labeled=True, w_features = phanded_dev )

  model.load(f'logs/{mod_name}_{language[:2]}_{rep}.pt')

  encs = model.get_encodings([encodings_train, handed_train], rep)
  torch.save(np.array(encs), f'logs/modelings/train_{task}_{mod_name}_{language[:2]}.pt')
  
  encs = model.get_encodings( [encodings_dev, handed_dev], rep)
  torch.save(np.array(encs), f'logs/modelings/test_{task}_{mod_name}_{language[:2]}.pt')
  print(f"{bcolors.OKCYAN}{bcolors.BOLD}Encodings Saved Successfully{bcolors.ENDC}")


if __name__ == '__main__':


  parameters = check_params(sys.argv[1:])

  lstm_hidden_size = parameters.lstm_size
  learning_rate, decay = parameters.lr,  parameters.decay
  splits = parameters.splits
  interm_layer_size = parameters.interm_layer
  max_length = parameters.ml
  mode = parameters.mode
  weight_path = parameters.wp
  batch_size = parameters.bs
  language = parameters.l
  epoches = parameters.epoches
  data_path = parameters.dp
  mode_weigth = parameters.tmode
  loss = parameters.loss
  metric = parameters.metric
  coef = parameters.rp
  ecnImp = parameters.ecnImp
  # filee = parameters.f
  test_path = parameters.dt
  phase = parameters.phase
  output = parameters.output
  up = parameters.up
  task = parameters.task
  rep = parameters.rep 

  if mode == 'encoder':

    prefix_path = '/content/drive/MyDrive/Profiling/logs'
    model_name = f'{prefix_path}/encoder_trans_{task}_{language[:2]}'
    if phase == 'train':
      '''
        Train Transformers based encoders BETo for spanish and BERTweet for English
      '''
      if os.path.exists(prefix_path) == False:
        os.system(f'mkdir {prefix_path}')
      labels, tweets_word, = read_data(os.path.join(data_path, language.lower()), trans=True)
      
      history = train_Encoder(model_name, data_path, language, mode_weigth, [tweets_word, labels], splits, epoches, batch_size, max_length, interm_layer_size, learning_rate, decay, 1, 0.1)
      plot_training(history[-1], model_name, 'acc')
      plot_training(history[-1], model_name)
    
    elif phase == 'encode':

      '''
        Get Encodings for each author's message from the Transformer based encoders
      '''
      weight_path = os.path.join(weight_path, f'{model_name}.pt')
      
      if os.path.isfile(f'{model_name}.pt') == False:
        print( f"{bcolors.FAIL}{bcolors.BOLD}ERROR: Weight path set unproperly{bcolors.ENDC}")
        exit(1)

      model = Encoder(interm_layer_size, max_length, language, mode_weigth)
      model.load(f'{model_name}.pt')
      if language[-1] == '_':
        model.transformer.load_adapter("logs/hate_adpt_{}".format(language[:2].lower()))
      
      tweets, _ = load_Profiling_Data(os.path.join(data_path, language[:2].lower()), False)
      preds = []
      encs = []
      batch_size = 200
      for i in tweets:
        e, _ = model.get_encodings(i, batch_size)
        encs.append(e)
        # preds.append(l)
      infosave = data_path.split("/")[-2:]
      torch.save(np.array(encs), f'{prefix_path}/{infosave[0]}_{infosave[1]}_encodings_{language[:2]}.pt')
      # torch.save(np.array(preds),f'logs/{}_pred_{}.pt'.format(phase, language[:2]))
      print(f"{bcolors.OKCYAN}{bcolors.BOLD}Encodings Saved Successfully{bcolors.ENDC}")

  if mode == 'CNN_LSTM_Encoder' :

    language = language.lower()
    emb_path = None
    if language == "en":
      emb_path = 'data/embeddings/glove_en_100d'
    elif language == "es": emb_path = 'data/embeddings/glove_es_200d'
    
    matrix, dic = read_embedding(emb_path)
   

    if phase == 'train':
      model_name = f'CNN_LSTM_ENC_{task}_{learning_rate}'
      # data_path = "../data/profiling/faker/train"
      model = SeqEncoder(language, matrix)
      labels, tweets_word, tweets_char, _, _, _ = read_data(os.path.join(data_path, language), dic)
      
      hist = train_Seq(model, [tweets_word, tweets_char, labels], language, model_name, splits, epoches, batch_size, lr = learning_rate,  decay=decay)
      plot_training(hist[-1], f'logs/{model_name}_{language}', 'acc')
      plot_training(hist[-1], f'logs/{model_name}_{language}')
    
    if phase == 'encode':
      model_name = f'CNN_LSTM_ENC_{task}'
      model = SeqEncoder(language, matrix)
      model.load(f'logs/{model_name}_{language}_1.pt')

      tweets, _ = load_Profiling_Data(os.path.join(data_path, language[:2].lower()), False)
      encs = []
      dicc = {' ': 0}
      for i in range(26):
          dicc[chr(i + 97)] = i + 1
      dicc['\''] = 27

      for i in tweets:
        tw, _, _ = translate_words(i, dic, 120)
        tc, _ = translate_char(i, dicc, 200)
        e = model.get_encodings([tw, tc], 200)
        encs.append(e)
      infosave = data_path.split("/")[-2:]
      torch.save(np.array(encs), f'logs/{infosave[0]}_{infosave[1]}_encodings_{language[:2].upper()}.pt')
      
      print(f"{bcolors.OKCYAN}{bcolors.BOLD}Encodings Saved Successfully as {infosave[0]}_{infosave[1]}_encodings_{language[:2]}.pt {bcolors.ENDC}")

  if mode == 'lstm':

    '''
      Train Train Att-LSTM
    ''' 
    if phase == 'train':

      encodings_train = None
      encodings_dev = None

      handed_train = None
      handed_dev = None
      phanded_train = None
      phanded_dev = None

      if 't' in rep:
        encodings_train = torch.load(f'logs/transformers/{task}_train_encodings_{language[:2]}.pt')
        encodings_dev = torch.load(f'logs/transformers/{task}_dev_encodings_{language[:2]}.pt')
      elif 'c' in rep:
        encodings_train = torch.load(f'logs/cnn_lstm/{task}_train_encodings_{language[:2]}.pt')
        encodings_dev = torch.load(f'logs/cnn_lstm/{task}_dev_encodings_{language[:2]}.pt')

      if 'h' in rep:
        phanded_train = f'logs/handcrafted/{task}_train_{language[:2].lower()}.json'
        phanded_dev = f'logs/handcrafted/{task}_dev_{language[:2].lower()}.json'

      _, _, labels_train, handed_train = load_Profiling_Data(f'{data_path}/train/{language.lower()}', labeled=True, w_features = phanded_train )
      _, _, labels_dev, handed_dev = load_Profiling_Data(f'{data_path}/dev/{language.lower()}', labeled=True, w_features = phanded_dev )

      history = train_classifier(task, rep, 'lstm', data_train = [encodings_train, labels_train], data_dev = [encodings_dev, labels_dev], 
                                  language = language, hfeaat={'train':handed_train, 'dev':handed_dev},splits = splits, epoches= epoches, batch_size = batch_size, 
                                  interm_layer_size = [interm_layer_size, 32, lstm_hidden_size], lr=learning_rate, decay=decay)
      
      plot_training(history[-1], f'logs/LSTM_{task}_{language}_{learning_rate}', 'acc')
      plot_training(history[-1], f'logs/LSTM_{task}_{language}_{learning_rate}')

    elif phase == 'encode':
      model = LSTMAtt_Classifier(interm_layer_size, 32, lstm_hidden_size, language, ('h' in rep))
      get_encodings(model, 'lstm')

    exit(0)


  if mode == 'cgnn':

    '''
      Train Train Graph Concolutional Neural Network
    ''' 
    from models.CGNN import train_GCNN, GCN, predicgcn
    if phase == 'train':

      encodings_train = None
      encodings_dev = None

      handed_train = None
      handed_dev = None
      phanded_train = None
      phanded_dev = None

      if 't' in rep:
        encodings_train = torch.load(f'logs/transformers/{task}_train_encodings_{language[:2]}.pt')
        encodings_dev = torch.load(f'logs/transformers/{task}_dev_encodings_{language[:2]}.pt')
      elif 'c' in rep:
        encodings_train = torch.load(f'logs/cnn_lstm/{task}_train_encodings_{language[:2]}.pt')
        encodings_dev = torch.load(f'logs/cnn_lstm/{task}_dev_encodings_{language[:2]}.pt')

      # if 'h' in rep:
      phanded_train = f'logs/handcrafted/{task}_train_{language[:2].lower()}.json'
      phanded_dev = f'logs/handcrafted/{task}_dev_{language[:2].lower()}.json'

      _, _, labels_train, handed_train = load_Profiling_Data(f'{data_path}/train/{language.lower()}', labeled=True, w_features = phanded_train )
      _, _, labels_dev, handed_dev = load_Profiling_Data(f'{data_path}/dev/{language.lower()}', labeled=True, w_features = phanded_dev )

      history = train_GCNN(rep, task, [encodings_train, handed_train, labels_train], [encodings_dev, handed_dev, labels_dev], language, splits = splits, epoches = epoches, batch_size = batch_size, hidden_channels = interm_layer_size, lr=learning_rate, decay=decay)
      plot_training(history[-1], f'logs/GNN_{task}_{language}_{learning_rate}', 'acc')
      plot_training(history[-1], f'logs/GNN_{task}_{language}_{learning_rate}')
    
    elif phase == 'encode':
      
      model = GCN(language, interm_layer_size, 64, handed_features = ('h' in rep))
      get_encodings(model, 'gcn')
    exit(0)
    
  if mode == 'Impostor':

    ''' 
      Classify the profiles with Impostors Method 
    '''

    encodings_train = torch.load(f'logs/modelings/train_{task}_{rep}_{language[:2]}.pt')
    encodings_dev = torch.load(f'logs/modelings/test_{task}_{rep}_{language[:2]}.pt')
  
    _, _, labels_train, _ = load_Profiling_Data(f'{data_path}/train/{language.lower()}', labeled=True, w_features = None )
    _, _, labels_dev, _ = load_Profiling_Data(f'{data_path}/dev/{language.lower()}', labeled=True, w_features = None )

    print('*'*50)
    print(f'   coef:{coef}  Encoder:{rep} Language: {language}')
    print('*'*50)
    model = None
    P_idx = list(np.argwhere(labels_train==1).reshape(-1))
    N_idx = list(np.argwhere(labels_train==0).reshape(-1))

    y_hat = K_Impostor(encodings_train[P_idx], encodings_train[N_idx], encodings_dev, checkp=coef, method=metric, model=model)
      
    print(f'Accuracy Test: {np.round(accuracy_score(labels_dev, y_hat), decimals=3)}')
    print(classification_report(labels_dev, y_hat, target_names=[f'No {task}', task],  digits=4, zero_division=1))

  if mode == 'svm':
    svm(task, language)