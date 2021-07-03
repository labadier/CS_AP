#!bin/bash

hs=64
tmode=online
python main.py -l EN -mode encoder -phase train -dp data/tasks/faker -ml 100 -tmode $tmode -bs 64  -epoches 12 -interm_layer $hs -lr 2e-5
python main.py -l EN -mode encoder -phase encode -dp data/profiling/faker/train -wp logs  -tmode $tmode -bs 300 -interm_layer $hs
python main.py -l EN -mode encoder -phase encode -dp data/profiling/faker/dev -wp logs  -tmode $tmode -bs 300 -interm_layer $hs

# python main.py -l EN  -mode Impostor -dp data/profiling/faker/train -rp 0.45 -metric cosine -up random -ecnImp transformer -dt data/profiling/faker/dev -output logs -interm_layer $hs
