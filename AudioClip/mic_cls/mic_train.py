import os
import sys
import glob

import librosa
import librosa.display


import numpy as np

import torch
import torchvision as tv
from torch import nn
import torch.utils.data as Data

import matplotlib.pyplot as plt
from ts_dataloader import *
from utils_mic import *

from PIL import Image
from IPython.display import Audio, display

sys.path.append(os.path.abspath(f'{os.getcwd()}/..'))

from model import AudioCLIP
from utils.transforms import ToTensor1D



def training(model, audio, target, optimizer, device, epochs):
    model.train()
    for i in range(epochs):
        
        print(f"Epoch {i+1}")
        #for audio, target in data_loader:
        audio = audio.to(device)
        #target = target.to(device)

        # calculate loss
        batch_indices = torch.arange(audio.shape[0], dtype=torch.int64, device=device)
        _, loss = model(audio=audio, text=target, batch_indices=batch_indices)
        if loss.ndim > 0:
            loss = loss.mean()
        #loss = loss_fn(prediction, target)
        #print(loss)
        loss.requires_grad = True
        # backpropagate error and update weights
        optimizer.zero_grad()
        loss.backward(retain_graph=False)
        optimizer.step(None)

        print(f"loss: {loss.item()}")
        pass
        print("---------------------------")
    print("Finished training")

def testing(model, test_loader, device):
    model.eval()
    #test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            #test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    #test_loss /= len(test_loader.dataset)

    print('Accuracy: {}/{} ({:.0f}%)\n'.format(
         correct, len(test_loader.dataset),
         100. * correct / len(test_loader.dataset)))


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

torch.set_grad_enabled(False)

MODEL_FILENAME = 'AudioCLIP-Full-Training.pt'
# derived from ESResNeXt
SAMPLE_RATE = 44100
# derived from CLIP
#IMAGE_SIZE = 224
#IMAGE_MEAN = 0.48145466, 0.4578275, 0.40821073
#IMAGE_STD = 0.26862954, 0.26130258, 0.27577711

LABELS = ['C', 'D', 'M', 'P']
classes = LABELS
limit = 5

train_csv = '/home/hsinhung/mic_acoustic/12class/data/train_full_mobile_clo_4th.csv'
dev_csv = '/home/hsinhung/mic_acoustic/12class/data/dev_full_mobile_clo_4th.csv'
test_csv = '/home/hsinhung/mic_acoustic/12class/data/test_full_mobile_clo_4th.csv'


print('loading microphone data')
train = load_data(train_csv)
dev = load_data(dev_csv)
test = load_data(test_csv)

if limit < 200:
    train = split(train, limit)

#print(train)

print ("=== Number of training data: {}".format(len(train)))
#print ("=== Number of test data: {}".format(len(test)))

x_train, y_train_4, y_train_18 = list(zip(*train))
x_dev, y_dev_4, y_dev_18 = list(zip(*dev))
x_test, y_test_4, y_test_18 = list(zip(*test))
x_test = np.array(x_test)
x_train = np.array(x_train)
x_dev = np.array(x_dev)



y_train = y_train_4
y_dev = y_dev_4
y_test = y_test_4

'''
cls2label = {label: i for i, label in enumerate(classes)}
num_classes = len(classes)

y_train = [cls2label[y] for y in y_train]
y_dev = [cls2label[y] for y in y_dev]
y_test = [cls2label[y] for y in y_test]
y_train = to_categorical(y_train, num_classes)
y_dev = to_categorical(y_dev, num_classes)
y_test = to_categorical(y_test, num_classes)
'''
#print(y_train)
x_train = torch.tensor(x_train)
y_train = [[label] for label in y_train]
#y_train = torch.tensor(y_train)
#print(y_train)
x_train = x_train[:,None,:]

'''
torch_data = Data.TensorDataset(x_train, y_train)
train_loader = Data.DataLoader(
        dataset = torch_data,
        batch_size = 64,
        shuffle = True
)
'''

text = [[label] for label in LABELS]

model = AudioCLIP(pretrained='../assets/'+MODEL_FILENAME).to(device)
print('==== ACLIP model loded')


#model = aclp
# disable all parameters
for p in model.parameters():
    p.requires_grad = False

# enable only audio-related parameters
for p in model.audio.parameters():
    p.requires_grad = True

# disable fbsp-parameters
for p in model.audio.fbsp.parameters():
    p.requires_grad = False

# disable logit scaling
model.logit_scale_ai.requires_grad = False
model.logit_scale_at.requires_grad = False

# add only enabled parameters to optimizer's list
param_groups = [
    {'params': [p for p in model.parameters() if p.requires_grad]}
]

# enable fbsp-parameters
for p in model.audio.fbsp.parameters():
    p.requires_grad = True

# enable logit scaling
model.logit_scale_ai.requires_grad = True
model.logit_scale_at.requires_grad = True

param_groups.append({
            'params': [
                p for p in model.audio.fbsp.parameters()
            ] + [
                model.logit_scale_ai,
                model.logit_scale_at
            ],
            'weight_decay': 0.0
        })


#loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(param_groups,
                                 lr=0.001)

num_params_total = sum(p.numel() for p in model.parameters())
num_params_train = sum(p.numel() for grp in optimizer.param_groups for p in grp['params'])

print('Total number of parameters: ', f'{num_params_total:,}')
print('Number of trainable parameters: ', f'{num_params_train:,}')
# train model
training(model, x_train, y_train, optimizer, device, 20)

#testing()


'''
audio_transforms = ToTensor1D()

image_transforms = tv.transforms.Compose([
    tv.transforms.ToTensor(),
    tv.transforms.Resize(IMAGE_SIZE, interpolation=Image.BICUBIC),
    tv.transforms.CenterCrop(IMAGE_SIZE),
    tv.transforms.Normalize(IMAGE_MEAN, IMAGE_STD)
])

paths_to_audio = sorted(glob.glob('music/*.wav'))

audio = list()
for path_to_audio in paths_to_audio:
    track, _ = librosa.load(path_to_audio, sr=SAMPLE_RATE, dtype=np.float32)

    # compute spectrograms using trained audio-head (fbsp-layer of ESResNeXt)
    # thus, the actual time-frequency representation will be visualized
    spec = aclp.audio.spectrogram(torch.from_numpy(track.reshape(1, 1, -1)))
    spec = np.ascontiguousarray(spec.numpy()).view(np.complex64)
    pow_spec = 10 * np.log10(np.abs(spec) ** 2 + 1e-18).squeeze()

    audio.append((track, pow_spec))

#print(audio)
print('===== audio loaded')



# AudioCLIP handles raw audio on input, so the input shape is [batch x channels x duration]
audio = torch.stack([audio_transforms(track.reshape(1, -1)) for track, _ in audio])
# standard channel-first shape [batch x channels x height x width]
#images = torch.stack([image_transforms(image) for image in images])
# textual input is processed internally, so no need to transform it beforehand
text = [[label] for label in LABELS]

# AudioCLIP's output: Tuple[Tuple[Features, Logits], Loss]
# Features = Tuple[AudioFeatures, ImageFeatures, TextFeatures]
# Logits = Tuple[AudioImageLogits, AudioTextLogits, ImageTextLogits]

((audio_features, _, _), _), _ = aclp(audio=audio)
#((_, image_features, _), _), _ = aclp(image=images)
((_, _, text_features), _), _ = aclp(text=text)

print('==== embedding output')

audio_features = audio_features / torch.linalg.norm(audio_features, dim=-1, keepdim=True)
#image_features = image_features / torch.linalg.norm(image_features, dim=-1, keepdim=True)
text_features = text_features / torch.linalg.norm(text_features, dim=-1, keepdim=True)

print('==== embedding normalized')

scale_audio_image = torch.clamp(aclp.logit_scale_ai.exp(), min=1.0, max=100.0)
scale_audio_text = torch.clamp(aclp.logit_scale_at.exp(), min=1.0, max=100.0)
scale_image_text = torch.clamp(aclp.logit_scale.exp(), min=1.0, max=100.0)


#logits_audio_image = scale_audio_image * audio_features @ image_features.T
logits_audio_text = scale_audio_text * audio_features @ text_features.T
#logits_image_text = scale_image_text * image_features @ text_features.T

print('==== embedding simularity computed')

print('\t\tFilename, Audio\t\t\tTextual Label (Confidence)', end='\n\n')

# calculate model confidence
confidence = logits_audio_text.softmax(dim=1)
for audio_idx in range(len(paths_to_audio)):
    # acquire Top-3 most similar results
    conf_values, ids = confidence[audio_idx].topk(3)

    # format output strings
    query = f'{os.path.basename(paths_to_audio[audio_idx]):>30s} ->\t\t'
    results = ', '.join([f'{LABELS[i]:>15s} ({v:06.2%})' for v, i in zip(conf_values, ids)])

    print(query + results)

'''
'''
model para: turn on audio head only
fine tuen loss? image? if none -> none
training, audio/target? how to output to map text?
'''