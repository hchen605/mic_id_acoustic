


for i in {1..10..1}
do
    python rir_2stage_train.py --nclass 1 --limit 5 --eps 200 --seed $i
    python rir_2stage_test.py --nclass 1 --limit 5 --unseen 0 --seed $i

done


'''
for i in {1..10..1}
do
    CUDA_VISIBLE_DEVICES=0 python rir_2stage_train.py --nclass 1 --limit 5 --eps 200 --seed $i
    CUDA_VISIBLE_DEVICES=0 python rir_2stage_test.py --nclass 1 --limit 5 --unseen 1 --seed $i

done
'''
