#!/bin/bash

hs=64
tmode=offline
# python main.py -l EN -mode encoder -phase train -dp data/tasks/faker -ml 100 -tmode $tmode -bs 12  -epoches 1 -interm_layer $hs -lr 2e-5
# # python main.py -l EN -mode encoder -phase encode -dp data/profiling/faker/train -wp logs  -tmode $tmode -bs 300 -interm_layer $hs
# # python main.py -l EN -mode encoder -phase encode -dp data/profiling/faker/dev -wp logs  -tmode $tmode -bs 300 -interm_layer $hs

# python main.py -l EN  -mode Impostor -dp data/profiling/faker/train -rp 0.35 -metric cosine -up random -ecnImp transformer -dt data/profiling/faker/dev -output logs -interm_layer $hs

# python main.py -l EN -mode fcnn -phase train -dp data/profiling/faker/train -interm_layer $hs -lr 1e-4 -epoches 20 -bs 32
# python main.py -l EN -mode fcnn -phase test -dp data/profiling/faker/dev -interm_layer $hs -lr 3e-4 -epoches 80 -bs 32
python main.py -l EN -mode fcnn -phase encode -dp data/profiling/faker/train -wp logs -bs 300 -interm_layer $hs
python main.py -l EN -mode fcnn -phase encode -dp data/profiling/faker/dev -wp logs -bs 300 -interm_layer $hs

# python main.py -mode lstm -phase train -dp data/profiling/faker/train -lr 1e-4 -epoch 64 -bs 32 -lstm_size 32 -decay 0
# python main.py -mode lstm -phase test -dp data/profiling/faker/dev -lr 1e-4 -epoch 64 -bs 32 -lstm_size 32 -decay 0
python main.py -mode lstm -phase encode -dp data/profiling/faker/train -lr 1e-4 -epoch 64 -bs 32 -lstm_size 32 -decay 0
python main.py -mode lstm -phase encode -dp data/profiling/faker/dev -lr 1e-4 -epoch 64 -bs 32 -lstm_size 32 -decay 0
