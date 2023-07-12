
CUDA_VISIBLE_DEVICES=0 python mic_all_train_teacher.py --nclass 1  --eps 200
CUDA_VISIBLE_DEVICES=0 python mic_all_test.py --nclass 1