# mic_id_acoustic

## Docker env setup

#### Step 1: build docker environment

```bash
docker build -t hchen605/mic_id_acoustic .
```

#### Step 2: Launch docker environment

```bash
docker run --shm-size=1g -v <dataset_path>:/home/speech -it --rm --privileged --gpus all -w /home/mic_id_acoustic hchen605/mic_id_acoustic:latest
```

Please place the dataset folders, crisp_record, crisp_phase3, Mobile_Recording, and all D1 close-field data in the same <dataset_path>.


*Env back-up: You can first follow the Docker flow shown in the GitHub repo. Below is the back-up flow for stand alone conda env setup with the `repr.yml` (AttCNN) and `clip.yml` (ESResNeXt).


## Microphone Classification

In the `./12class` directory you will find the codes. In `./data` directory you will find the .csv files indicating all train/dev/test file path from dataset folder. Make sure to replace the directory path to your local path (if needed). In the `./fcnn` you will find the train and test script independently.

1. All microphone classification including mobile phones

Please refer to `run_mic_all.sh` script. The setting is the same as D1.
Note that during training stage, the model weights will be saved in your local path. 

![image](https://user-images.githubusercontent.com/78195585/173097212-364f7ee1-29ab-4089-a574-a5c9e7d196ef.png)

`limit5` means 5 data per microphone (Train = 90). The result is the average of seed1~10. 

Note: dataset folder for small room and large room.

Data recorded in small room (close-field):

- 12 mic classes: The same as D1 deliverable.
- Mobile data: in the Mobile_Recording folder, for each mobile phone folder, you will find the recordings in `clo_*` folder.

Data recorded in large room (different distances):

- 12 mic classes: In `crisp_record` folder with distance specified. `crisp_phase3`
- Mobile data: in the Mobile_Recording folder, for each mobile phone folder, you will find the recordings in folder with distance specified. Ex: 5m_train/test

### ESResNe(X)t-fbsp

You may find the code in the ```./ESResNeXt-fbsp``` directory. To setup the parameters, please find the in ```./protocols/mic_classification/*.json``` files. You can set the Dataset arg ‘limit’ to decide the low resource number in the training .json file.


##### Train

```bash
wget https://github.com/AndreyGuzhov/ESResNeXt-fbsp/releases/download/v0.1/ESResNeXtFBSP_AudioSet.pt

visdom -port <PORT|8097>
cd ESResNeXt-fbsp
python main.py --visdom-port <PORT|8097> --config protocols/mic_classificatoin/esresnextfbsp-mc-ptinas-cv1.json
```

##### Test

```bash
python main.py --pretrained <path> --config protocols/mic_classificatoin/esresnextfbsp-mc-ptinas-test-cv1.json
```
You can load the weight (from provided or result from training) at the Model args ‘pretrained’. The pre-trained weights can be found in `./ESResNeXt-fbsp/weights/`. For example, the `./mic_18class_5` means the weight for 18-class evaluation with low-resource 5 data per class.


### 4 class evaluation (mobile data as the 4th class)

Train: train/dev_full_mobile_clo_4th.csv
Test: test_full_mobile_clo_4th.csv

Unseen device test (4 class):

Train: train/dev_full_mobile_clo_4th_sp1.csv
Test: test_full_mobile_clo_4th_sp1.csv

### Unseen Device Evaluation

Please run the following command:
```bash
python mic_all_train_pretrain_imloss.py --nclass 0 --limit 400 --eps 100
python mic_all_test.py --nclass 0 --limit 400
```

The first one is the training process which will apply IM loss and include updated pseudo labels to fine tune the model. The second command is the evaluation for unseen devices.

Note: Please check in `./data/` for the relevant .csv files. The model is pre-trained by split 1 configuration, referring to the `train/dev_ full_mobile_4th_sp1.csv`. Then the model is fine-tuned with IM loss and the updated pseudo labels (`./data/ood/pseudo_update_4.py`). And then evaluated with `test_full_mobile_4th_sp1.csv` to get the refined result.

The pre-trained weight for the domain adaptation can be found in `./weight/domainAdapt_imloss/`. You may apply this weight to evaluate the unseen device data.(The raw pre-trained weight for 4 class without domain adaptation is in `./split1_pretrain_4class`.)

### Unseen room/distance Evaluation

Please run the following command:
```bash
source run_teacher_student.sh
```
This will automatically run the `mic_all_train_teacher.py` to train the teacher-student model and run `mic_all_test.py` for evaluation.

### Out-of-Domain Detection (OOD detection)

#### AttCNN:

To calculate the AUROC of ID and OOD, you can run the ```./fcnn/mic_all_test_abstention_distance.py``` with the command: ```python mic_all_test_abstention_distance.py --nclass 1```

You can modify the ‘test_csv’ and ‘test_csv_2’ for Config2 ID and OOD. For example, the setting below is for training P1 as ID and P2-4 as OOD, then tested with P1 and P5-6 for unseen ID and OOD evaluation.

```bash
test_csv = '../data/ood/test_full_mobile_clo_4th_abstention_p1_dev_.csv'
test_csv_2 = '../data/ood/test_full_mobile_clo_4th_abstention_p_dev.csv'
```

Note: Please check in `./data/ood` for the collected .csv file. Take config.2 for example, the training data is the `train_full_mobile_4th_abstention_p_trial_p1_dev.csv`, which includes the P1 as the ID data and P2-4 as OOD data. The validation set is of the `dev_full_mobile_4th_abstention_p_trial_p1_dev.csv`, also including P1 as the ID data and P2-4 as OOD data. The test data is of the `test_full_mobile_4th_abstention_p_trial_p1_dev_.csv`, which is the test ID data. And `test_full_mobile_4th_sp1_p.csv` is the test OOD data (P5-6). You can also find the data collection of config.1 with the post-fix like ‘p1p2’ and ‘p3p4’. 

The pre-trained weights can be found in the ./fcnn/weight/ directory. For example, the ./ood_p1_id is the weight for P1 split for config.2.


## Room size estimation

### AttCNN

In the `./12class/fcnn` directory you will find the codes for D2. In `./data` directory you will find the .csv files indicating all train/dev/test file path from dataset folder. Make sure to replace the directory name to your local path. In the `./fcnn` you will find the train and test script independently. For example, `room_train.py` and `room_test.py`. Additional `run_room.sh` is provided for your reference.

The example script is shown as below.

![image](https://user-images.githubusercontent.com/78195585/173098483-d6b8b549-be02-4034-80f7-ff0dadda103e.png)

The script automatedly call the train/test dataset needed. 
Note that in order to run unseen scenario, you may change –unseen to 1.



#### Unseen room evaluation with real RIR augmentation

Train: train/dev_room.csv
Test: test_room_unseen.csv

Note: D3 dataset

For D3 unseen room size evaluation, real RIR simulated data is used. Please check in `./speech/rir_large_$` for the simulated large room data, `./speech/rir_medium_$` for the simulated medium room data, and `./speech/rir_small_$` for the simulated small room data. 

Note: Real RIR simulation

The sample real RIR simulation tool is shown in `rir_gen_room.py`. The real RIR adopted for different room sizes are shown in `./RIR_room`. You can follow the sample to simulated new data with your own recordings.

Since some RIR will last longer period, the output audio length will not be consistent for different RIR. In this case, we evaluate the model with the data retrieved from the start until, for example, 3 seconds. 

### ESResNeXt:

You may find the code in the `./ESResNeXt-fbsp directory`. To setup the parameters, please find the file in `./protocols/room_classification/*.json` files. 


You can directly run the command to get the result: 

```CUDA_VISIBLE_DEVICES=$N python main.py –config protocols/room_classification/esresnextfbsp-room-test.json```

The pre-trained weight can be found in `./ESResNeXt-fbsp/weights/room/best.pth`
You may load this weight in the test json file and run the evaluation directly.

## Distance Prediction

For new update, please refer to the new repo specific for distance prediction.
https://github.com/hchen605/mic_source_dist

====

In the `./microphone_classification/12class/fcnn` directory you will find the codes for D2. In `../data` directory you will find the .csv files indicating all train/dev/test file path from dataset folder. Make sure to replace the directory name to your local path. In the `./fcnn` you will find the train and test script independently. For example, `dist_train.py` and `dist_test.py`. Additional `run_dist.sh` is provided for your reference.

The example script is shown as below.

![image](https://user-images.githubusercontent.com/78195585/173107938-2671e473-ccf8-4a33-a85d-bf4179d6959d.png)

Note: dataset folder
Please check ./crisp_record for real recordings for 5m, 7m, 9m.
And please check ./crips_record_mixed for data mixed with different DoA. Some simulated data with different locations are from `rir_*m_loc_*` in ./Dataset. The data will be call based on .csv file used.
