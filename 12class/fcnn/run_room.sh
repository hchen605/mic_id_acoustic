

for i in {1..10..1}
do
    python room_train.py --target 1 --nclass 1 --limit 50 --eps 100 --seed $i
    python room_test.py --target 1 --nclass 1 --limit 50 --unseen 0 --seed $i
done


for i in {1..10..1}
do
    python room_train.py --target 1 --nclass 1 --limit 100 --eps 100 --seed $i
    python room_test.py --target 1 --nclass 1 --limit 100 --unseen 0 --seed $i
done

for i in {1..10..1}
do
    python room_train.py --target 1 --nclass 1 --limit 50 --eps 100 --seed $i
    python room_test.py --target 1 --nclass 1 --limit 50 --unseen 1 --seed $i
done


for i in {1..10..1}
do
    python room_train.py --target 1 --nclass 1 --limit 100 --eps 100 --seed $i
    python room_test.py --target 1 --nclass 1 --limit 100 --unseen 1 --seed $i
done
