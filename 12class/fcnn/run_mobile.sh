

for i in {1..10..1}
do
    python mic_all_train.py --nclass 0 --limit 5 --eps 100 --seed $i || exit 1
    python mic_all_test.py --nclass 0 --limit 5 --seed $i || exit 1
done

for i in {1..10..1}
do
    python mic_all_train.py --nclass 1 --limit 5 --eps 100 --seed $i || exit 1
    python mic_all_test.py --nclass 1 --limit 5 --seed $i || exit 1
done

for i in {1..10..1}
do
    python eval_fcnn_mobile.py --limit 5 --seed $i || exit 1
done


