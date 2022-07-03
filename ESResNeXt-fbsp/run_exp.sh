
VISDOM_PORT=8888

visdom -port $VISDOM_PORT -logging_level WARNING > /dev/null &
mkdir -p log

. ./clean.sh
for i in `seq 1 20`; do
    rm -rf weights/MicClassification_PTINAS_ESRNXFBSP-MC
    python main.py \
        --visdom-port $VISDOM_PORT \
        --config protocols/mic_classificatoin/esresnextfbsp-mc-ptinas-cv1.json
    [ ${PIPESTATUS[0]} -eq 1 ] && pkill visdom && exit 1
    model=`ls weights/MicClassification_PTINAS_ESRNXFBSP-MC`
    python main.py --pretrained weights/MicClassification_PTINAS_ESRNXFBSP-MC/$model \
        --config protocols/mic_classificatoin/esresnextfbsp-mc-ptinas-test-cv1.json \
            | tee log/$i.log
    [ ${PIPESTATUS[0]} -eq 1 ] && pkill visdom && exit 1
done

pkill visdom
