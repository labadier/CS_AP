#!/bin/bash

hs=64
tmode=offline
# clear

############################################# Train CNN_LSTM_Encoder ############################################
# python main.py -l ES -mode CNN_LSTM_Encoder -phase train -dp data/profiling/faker/train -lr 2e-3 -epoch 12 -bs 64 -decay 0 -task faker
# python main.py -l EN -mode CNN_LSTM_Encoder -phase train -dp data/profiling/faker/train -lr 2e-3 -epoch 12 -bs 64 -decay 0 -task faker
# python main.py -l ES -mode CNN_LSTM_Encoder -phase train -dp data/profiling/hater/train -lr 2e-3 -epoch 12 -bs 64 -decay 0 -task hater
# python main.py -l EN -mode CNN_LSTM_Encoder -phase train -dp data/profiling/hater/train -lr 2e-3 -epoch 12 -bs 64 -decay 0 -task hater
# python main.py -l ES -mode CNN_LSTM_Encoder -phase train -dp data/profiling/bot/train -lr 2e-3 -epoch 12 -bs 64 -decay 0 -task bot
# python main.py -l EN -mode CNN_LSTM_Encoder -phase train -dp data/profiling/bot/train -lr 2e-3 -epoch 12 -bs 64 -decay 0 -task bot

# python main.py -l ES -mode CNN_LSTM_Encoder -phase train -dp data/profiling/faker/train -lr 1e-3 -epoch 12 -bs 64 -decay 0 -task faker
# python main.py -l EN -mode CNN_LSTM_Encoder -phase train -dp data/profiling/faker/train -lr 1e-3 -epoch 12 -bs 64 -decay 0 -task faker
# python main.py -l ES -mode CNN_LSTM_Encoder -phase train -dp data/profiling/hater/train -lr 1e-3 -epoch 12 -bs 64 -decay 0 -task hater
# python main.py -l EN -mode CNN_LSTM_Encoder -phase train -dp data/profiling/hater/train -lr 1e-3 -epoch 12 -bs 64 -decay 0 -task hater
# python main.py -l ES -mode CNN_LSTM_Encoder -phase train -dp data/profiling/bot/train -lr 1e-3 -epoch 12 -bs 64 -decay 0 -task bot
# python main.py -l EN -mode CNN_LSTM_Encoder -phase train -dp data/profiling/bot/train -lr 1e-3 -epoch 12 -bs 64 -decay 0 -task bot

# python main.py -l ES -mode CNN_LSTM_Encoder -phase train -dp data/profiling/faker/train -lr 15e-4 -epoch 12 -bs 64 -decay 0 -task faker
# python main.py -l EN -mode CNN_LSTM_Encoder -phase train -dp data/profiling/faker/train -lr 15e-4 -epoch 12 -bs 64 -decay 0 -task faker
# python main.py -l ES -mode CNN_LSTM_Encoder -phase train -dp data/profiling/hater/train -lr 15e-4 -epoch 12 -bs 64 -decay 0 -task hater
# python main.py -l EN -mode CNN_LSTM_Encoder -phase train -dp data/profiling/hater/train -lr 15e-4 -epoch 12 -bs 64 -decay 0 -task hater
# python main.py -l ES -mode CNN_LSTM_Encoder -phase train -dp data/profiling/bot/train -lr 15e-4 -epoch 12 -bs 64 -decay 0 -task bot
# python main.py -l EN -mode CNN_LSTM_Encoder -phase train -dp data/profiling/bot/train -lr 15e-4 -epoch 12 -bs 64 -decay 0 -task bot

# python main.py -l ES -mode CNN_LSTM_Encoder -phase train -dp data/profiling/faker/train -lr 3e-3 -epoch 12 -bs 64 -decay 0 -task faker
# python main.py -l EN -mode CNN_LSTM_Encoder -phase train -dp data/profiling/faker/train -lr 3e-3 -epoch 12 -bs 64 -decay 0 -task faker
# python main.py -l ES -mode CNN_LSTM_Encoder -phase train -dp data/profiling/hater/train -lr 3e-3 -epoch 12 -bs 64 -decay 0 -task hater
# python main.py -l EN -mode CNN_LSTM_Encoder -phase train -dp data/profiling/hater/train -lr 3e-3 -epoch 12 -bs 64 -decay 0 -task hater
# python main.py -l ES -mode CNN_LSTM_Encoder -phase train -dp data/profiling/bot/train -lr 3e-3 -epoch 12 -bs 64 -decay 0 -task bot
# python main.py -l EN -mode CNN_LSTM_Encoder -phase train -dp data/profiling/bot/train -lr 3e-3 -epoch 12 -bs 64 -decay 0 -task bot




# python main.py -l ES -mode CNN_LSTM_Encoder -phase encode -dp data/profiling/faker/dev -lr 2e-3 -epoch 12 -bs 64 -decay 0
# python main.py -l EN -mode CNN_LSTM_Encoder -phase encode -dp data/profiling/faker/dev -lr 2e-3 -epoch 12 -bs 64 -decay 0
# python main.py -l ES -mode CNN_LSTM_Encoder -phase encode -dp data/profiling/faker/train -lr 2e-3 -epoch 12 -bs 64 -decay 0
# python main.py -l EN -mode CNN_LSTM_Encoder -phase encode -dp data/profiling/faker/train -lr 2e-3 -epoch 12 -bs 64 -decay 0

############################################# Train Transformer Encoder ############################################
python main.py -l ES -mode encoder -phase train -task faker -dp data/tasks/faker -ml 120 -tmode $tmode -bs 12  -epoches 1 -interm_layer $hs -lr 2e-5
python main.py -l ES -mode encoder -phase train -task bot -dp data/tasks/bot -ml 120 -tmode $tmode -bs 12  -epoches 1 -interm_layer $hs -lr 2e-5
python main.py -l ES -mode encoder -phase train -task hater -dp data/tasks/hater -ml 120 -tmode $tmode -bs 12  -epoches 1 -interm_layer $hs -lr 2e-5

python main.py -l EN -mode encoder -phase train -task faker -dp data/tasks/faker -ml 120 -tmode $tmode -bs 12  -epoches 1 -interm_layer $hs -lr 2e-5
python main.py -l EN -mode encoder -phase train -task bot -dp data/tasks/bot -ml 120 -tmode $tmode -bs 12  -epoches 1 -interm_layer $hs -lr 2e-5
python main.py -l EN -mode encoder -phase train -task hater -dp data/tasks/hater -ml 120 -tmode $tmode -bs 12  -epoches 1 -interm_layer $hs -lr 2e-5
















# # python main.py -l EN -mode encoder -phase encode -dp data/profiling/faker/train -wp logs  -tmode $tmode -bs 300 -interm_layer $hs
# # python main.py -l EN -mode encoder -phase encode -dp data/profiling/faker/dev -wp logs  -tmode $tmode -bs 300 -interm_layer $hs

# python main.py -l EN  -mode Impostor -dp data/profiling/faker/train -rp 0.35 -metric cosine -up random -ecnImp transformer -dt data/profiling/faker/dev -output logs -interm_layer $hs

# python main.py -l EN -mode fcnn -phase train -dp data/profiling/faker/train -interm_layer $hs -lr 1e-4 -epoches 20 -bs 32
# python main.py -l EN -mode fcnn -phase test -dp data/profiling/faker/dev -interm_layer $hs -lr 3e-4 -epoches 80 -bs 32
# python main.py -l EN -mode fcnn -phase encode -dp data/profiling/faker/train -wp logs -bs 300 -interm_layer $hs
# python main.py -l EN -mode fcnn -phase encode -dp data/profiling/faker/dev -wp logs -bs 300 -interm_layer $hs


# python main.py -l EN -mode lstm -phase train -dp data/profiling/faker/train -lr 1e-4 -epoch 20 -bs 64 -lstm_size 32 -interm_layer 64 -decay 0
# python main.py -l EN -mode lstm -phase test -dp data/profiling/faker/dev -lr 1e-4 -bs 32 -lstm_size 32 -interm_layer 64 -decay 0
# python main.py -l ES -mode lstm -phase train -dp data/profiling/faker/train -lr 1e-4 -epoch 20 -bs 64 -lstm_size 32 -interm_layer 64 -decay 0
# python main.py -l ES -mode lstm -phase test -dp data/profiling/faker/dev -lr 1e-4 -bs 32 -lstm_size 32 -interm_layer 64 -decay 0

# python main.py -mode lstm -phase encode -dp data/profiling/faker/train -lr 1e-4 -epoch 64 -bs 32 -lstm_size 32 -decay 0
# python main.py -mode lstm -phase encode -dp data/profiling/faker/dev -lr 1e-4 -epoch 64 -bs 32 -lstm_size 32 -decay 0
