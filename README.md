# mic_id_acoustic

Please install the environment with the `repr.yml`

## Microphone Classification

In the `./12class` directory you will find the codes. In `./data` directory you will find the .csv files indicating all train/dev/test file path from dataset folder. Make sure to replace the directory path to your local path (if needed). In the `./fcnn` you will find the train and test script independently.

1. All microphone classification including mobile phones

Please refer to `run_mobile.sh` script. The setting is the same as D1.
You can also run `eval_fcnn_mobile.py` to evaluate two-stage 18 class results as shown below. Note that during training stage, the model weights will be saved in your local path. 

![image](https://user-images.githubusercontent.com/78195585/173097212-364f7ee1-29ab-4089-a574-a5c9e7d196ef.png)

`limit5` means 5 data per microphone (Train = 90). The result is the average of seed1~10. 

Note: dataset folder for small room and large room.

Data recorded in small room:

- 12 classes: The same as D1 deliverable.
- Mobile: in the Mobile_Recording folder, for each mobile phone folder, you will find the recordings in `clo_*` folder.

Data recorded in large room:

- 12 classes: in `crisp_record` folder with distance specified.
- Mobile: in the Mobile_Recording folder, for each mobile phone folder, you will find the recordings in folder with distance specified. Ex: 5m_train/test

### #### ESResNe(X)t-fbsp

```
wget https://github.com/AndreyGuzhov/ESResNeXt-fbsp/releases/download/v0.1/ESResNeXtFBSP_AudioSet.pt

visdom -port <PORT|8097>
cd ESResNeXt-fbsp
python main.py --visdom-port <PORT|8097> --config protocols/mic_classificatoin/esresnextfbsp-mc-ptinas-cv1.json
```

### Most recent experiments: 4 class evaluation

Train: train/dev_full_mobile_clo_4th.csv
Test: test_full_mobile_clo_4th.csv

Unseen device test:

Train: train/dev_full_mobile_clo_4th_sp1.csv
Test: test_full_mobile_clo_4th_sp1.csv

## Room size estimation

In the `./12class/fcnn` directory you will find the codes for D2. In `./data` directory you will find the .csv files indicating all train/dev/test file path from dataset folder. Make sure to replace the directory name to your local path. In the `./fcnn` you will find the train and test script independently. For example, `room_train.py` and `room_test.py`. Additional `run_room.sh` is provided for your reference.

The example script is shown as below.

![image](https://user-images.githubusercontent.com/78195585/173098483-d6b8b549-be02-4034-80f7-ff0dadda103e.png)

The script automatedly call the train/test dataset needed. 
Note that in order to run unseen scenario, you may change –unseen to 1.

Note: dataset folder

- Simulated medium room data: in the ./rir folder. 
  ⋅⋅⋅* Simulated in 6.5x6.5x3 room size
  ⋅⋅⋅* The distance is fixed in the center with 3m distance
- Small room data: the same as D1 deliverable 
- Large room data: in the ./crisp_record folder

### Most recent experiments: Unseen room evaluation with real RIR augmentation

Train: train/dev_room.csv
Test: test_room_unseen.csv

## Distance Prediction

In the `./microphone_classification/12class/fcnn` directory you will find the codes for D2. In `../data` directory you will find the .csv files indicating all train/dev/test file path from dataset folder. Make sure to replace the directory name to your local path. In the `./fcnn` you will find the train and test script independently. For example, `dist_train.py` and `dist_test.py`. Additional `run_dist.sh` is provided for your reference.

The example script is shown as below.

![image](https://user-images.githubusercontent.com/78195585/173107938-2671e473-ccf8-4a33-a85d-bf4179d6959d.png)

Note: dataset folder
Please check ./crisp_record for real recordings for 5m, 7m, 9m.
And please check ./crips_record_mixed for data mixed with different DoA. Some simulated data with different locations are from `rir_*m_loc_*` in ./Dataset. The data will be call based on .csv file used.
