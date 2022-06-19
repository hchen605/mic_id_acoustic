
mkdir -p log

for nclass in `seq 0 1`; do
    for i in {1..10..1}; do
        python mic_all_train.py \
            --nclass $nclass \
            --limit 5 \
            --eps 100 \
            --seed $i 2>&1 |
        tee log/train.${nclass}class.seed$i.log; [ ${PIPESTATUS[0]} -eq 0 ] || exit 1

        python mic_all_test.py \
            --nclass $nclass \
            --limit 5 \
            --seed $i 2>&1 |
        tee log/test.${nclass}class.seed$i.log; [ ${PIPESTATUS[0]} -eq 0 ] || exit 1
    done
done

# for i in {1..10..1}; do
#     python eval_fcnn_mobile.py \
#         --limit 5 \
#         --seed $i 2>&1 | tee log/2stage.eval.seed$i.log; [ ${PIPESTATUS[0]} -eq 0 ] || exit 1
# done

