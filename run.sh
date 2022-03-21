# MyModel, voc
# pretrain
python3 run.py --purpose train --pretrain --dataset voc --which_gpu 1

# train
python3 run.py --purpose train --dataset voc --which_gpu 1

# eval
python3 run.py --eval --dataset voc --which_gpu 1