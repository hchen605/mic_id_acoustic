

for i in {1..10..1}
do
    CUDA_VISIBLE_DEVICES=0 python mic_all_train.py --nclass 0 --limit 5 --eps 100 --seed $i 
    CUDA_VISIBLE_DEVICES=0 python mic_all_test.py --nclass 0 --limit 5 --seed $i 
done

for i in {1..10..1}
do
    CUDA_VISIBLE_DEVICES=0 python mic_all_train.py --nclass 1 --limit 5 --eps 100 --seed $i 
    CUDA_VISIBLE_DEVICES=0 python mic_all_test.py --nclass 1 --limit 5 --seed $i 
done

for i in {1..10..1}
do
    CUDA_VISIBLE_DEVICES=0 python eval_fcnn_mobile.py --limit 5 --seed $i 
done


