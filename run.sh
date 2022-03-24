# MyModel, voc
# pretrain
python3 run.py --purpose train --pretrain --which_gpu 0 --model Completer --dataset Caltech


# train
python3 run.py --purpose train --which_gpu 0 --model Completer --dataset Caltech

# eval
python3 run.py --eval --which_gpu 0 --model Completer --dataset Caltech