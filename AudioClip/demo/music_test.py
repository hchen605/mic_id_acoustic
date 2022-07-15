import os
import sys
sys.path.append("..")
import glob

import librosa
import librosa.display

import simplejpeg
import numpy as np

import torch
import torchvision as tv

import matplotlib.pyplot as plt

from PIL import Image
from IPython.display import Audio, display

sys.path.append(os.path.abspath(f'{os.getcwd()}/..'))

from model import AudioCLIP
from utils.transforms import ToTensor1D


torch.set_grad_enabled(False)

MODEL_FILENAME = 'AudioCLIP-Full-Training.pt'
# derived from ESResNeXt
SAMPLE_RATE = 44100
# derived from CLIP
IMAGE_SIZE = 224
IMAGE_MEAN = 0.48145466, 0.4578275, 0.40821073
IMAGE_STD = 0.26862954, 0.26130258, 0.27577711

LABELS = ['piano', 'guitar', 'violin', 'flute', 'singing', 'trumpet']


#aclp = AudioCLIP(pretrained=f'../assets/{MODEL_FILENAME}')
aclp = AudioCLIP(pretrained='../assets/'+MODEL_FILENAME)
print('==== ACLIP model loded')


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
'''
fig, axes = plt.subplots(2, len(audio), figsize=(20, 5), dpi=100)

for idx in range(len(audio)):
    track, pow_spec = audio[idx]

    # draw the waveform
    librosa.display.waveplot(track, sr=SAMPLE_RATE, ax=axes[0, idx], color='k')
    # show the corresponding power spectrogram
    axes[1, idx].imshow(pow_spec, origin='lower', aspect='auto', cmap='gray', vmin=-180.0, vmax=20.0)

    # modify legend
    axes[0, idx].set_title(os.path.basename(paths_to_audio[idx]))
    axes[0, idx].set_xlabel('')
    axes[0, idx].set_xticklabels([])
    axes[0, idx].grid(True)
    axes[0, idx].set_ylim(bottom=-1, top=1)

    axes[1, idx].set_xlabel('Time (s)')
    axes[1, idx].set_xticks(np.linspace(0, pow_spec.shape[1], len(axes[0, idx].get_xticks())))
    axes[1, idx].set_xticklabels([f'{tick:.1f}' if tick == int(tick) else '' for tick in axes[0, idx].get_xticks()])
    axes[1, idx].set_yticks(np.linspace(0, pow_spec.shape[0] - 1, 5))

axes[0, 0].set_ylabel('Amplitude')
axes[1, 0].set_ylabel('Filter ID')

plt.show()
plt.close(fig)

for idx, path in enumerate(paths_to_audio):
    print(os.path.basename(path))
    display(Audio(audio[idx][0], rate=SAMPLE_RATE, embed=True))

print('==== image loaded')

'''

paths_to_images = glob.glob('images/*.jpg')

images = list()
for path_to_image in paths_to_images:
    with open(path_to_image, 'rb') as jpg:
        image = simplejpeg.decode_jpeg(jpg.read())
        images.append(image)

print('==== image loaded')

# AudioCLIP handles raw audio on input, so the input shape is [batch x channels x duration]
audio = torch.stack([audio_transforms(track.reshape(1, -1)) for track, _ in audio])
# standard channel-first shape [batch x channels x height x width]
images = torch.stack([image_transforms(image) for image in images])
# textual input is processed internally, so no need to transform it beforehand
text = [[label] for label in LABELS]

#print(audio.size(dim=0))
#print(audio.size(dim=1))
#print(audio.size(dim=2))
# AudioCLIP's output: Tuple[Tuple[Features, Logits], Loss]
# Features = Tuple[AudioFeatures, ImageFeatures, TextFeatures]
# Logits = Tuple[AudioImageLogits, AudioTextLogits, ImageTextLogits]

((audio_features, _, _), _), _ = aclp(audio=audio)
((_, image_features, _), _), _ = aclp(image=images)
((_, _, text_features), _), _ = aclp(text=text)

#print(audio_features.shape) [14, 1024]
#print(text_features.shape) [6, 1024]

print('==== embedding output')

audio_features = audio_features / torch.linalg.norm(audio_features, dim=-1, keepdim=True)
image_features = image_features / torch.linalg.norm(image_features, dim=-1, keepdim=True)
text_features = text_features / torch.linalg.norm(text_features, dim=-1, keepdim=True)

print('==== embedding normalized')

scale_audio_image = torch.clamp(aclp.logit_scale_ai.exp(), min=1.0, max=100.0)
scale_audio_text = torch.clamp(aclp.logit_scale_at.exp(), min=1.0, max=100.0)
scale_image_text = torch.clamp(aclp.logit_scale.exp(), min=1.0, max=100.0)


logits_audio_image = scale_audio_image * audio_features @ image_features.T
logits_audio_text = scale_audio_text * audio_features @ text_features.T
logits_image_text = scale_image_text * image_features @ text_features.T

#print(logits_audio_text.shape) [14, 6]

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
Filename, Audio                 Textual Label (Confidence)

                     flute.wav ->                         flute (28.24%),          guitar (27.15%),           piano (18.87%)
              flute_guitar.wav ->                         flute (30.06%),         trumpet (25.78%),          guitar (23.48%)
                  guitar_1.wav ->                         piano (34.29%),          guitar (33.18%),         trumpet (14.06%)
                  guitar_2.wav ->                        guitar (99.10%),         singing (00.26%),          violin (00.23%)
                  guitar_3.wav ->                        guitar (86.36%),           piano (04.92%),         trumpet (03.65%)
                  guitar_4.wav ->                        guitar (98.03%),         trumpet (00.69%),           flute (00.44%)
                   piano_1.wav ->                         piano (48.33%),          guitar (24.02%),           flute (15.92%)
                   piano_2.wav ->                        guitar (51.75%),         trumpet (28.93%),           flute (10.16%)
                 singing_1.wav ->                       trumpet (85.56%),          violin (08.27%),           flute (02.88%)
                 singing_2.wav ->                       singing (92.26%),           flute (04.28%),          guitar (01.46%)
                 trumpet_1.wav ->                       trumpet (95.67%),           flute (02.29%),          guitar (01.17%)
                 trumpet_2.wav ->                       trumpet (95.63%),          violin (03.31%),          guitar (00.54%)
                  violin_1.wav ->                        violin (33.86%),           piano (25.32%),         trumpet (17.98%)
                  violin_2.wav ->                         piano (78.37%),          violin (20.03%),           flute (00.77%)
'''