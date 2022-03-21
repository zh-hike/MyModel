# MyModel, voc
# pretrain
python3 run.py --purpose train --pretrain --dataset mnist --which_gpu 1

# train
python3 run.py --purpose train --dataset mnist --which_gpu 1

# eval
python3 run.py --eval --dataset mnist --which_gpu 1