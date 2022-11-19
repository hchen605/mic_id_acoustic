

for i in {1..10..1}
do
    python mic_all_train.py --nclass 0 --limit 100 --eps 100 --seed $i 
    python mic_all_test.py --nclass 0 --limit 100 --seed $i 
done

for i in {1..10..1}
do
    python mic_all_train.py --nclass 1 --limit 100 --eps 100 --seed $i 
    python mic_all_test.py --nclass 1 --limit 100 --seed $i 
done

