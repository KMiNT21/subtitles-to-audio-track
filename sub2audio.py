# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # The main use case:
#
# ## You have video with non-English audio, but you have English subtitles (or going to prepare). Now you are ready to generate new English audio-track for this video.
#
#
# This notebook working steps:
#
# 1) Download captions for video id, save forever to local file, so, if captions changed, delete local pickle-file and repeat.
# 2) Parse and clean text from captions
# 3) Split text to sentences and synthesize WAV files for each sentence by **Mozilla TTS**.
# 4) Arrange start point of each audio by matching text with subtitles. Visualize subtitles before and after arrangement by heat-map image. Rows = minutes. Cols = seconds. Numbers in cells = numbers of audio segment (check metadata.csv)
# 5) Concatenate all audio segment (and background music if not disabled) in one wav audio track.
# 6) *Replace audio track in local video file.* Deprecated: you should replace audio-track in any video editor manually.

# %% [markdown]
# # Settings

# %%
import os
from pathlib import Path

youtube_video_id = 'ImdWoHviA0k'

# folder where additional folders will be created for every youtube_video_id
data_folder_root = Path('s:/temp/data')
# data_folder_root = Path().absolute()/'caps2audio'  # uncomment this to use current folder
data_folder = data_folder_root/youtube_video_id

# set your path to .json after installing Coqui-AI TTS by pip
tss_models_json_path = "C:/Python/Python39/Lib/site-packages/TTS/.models.json"

USE_CUDA = True

os.makedirs(data_folder, exist_ok=True)
os.makedirs(data_folder/'wavs', exist_ok=True)


# %% [markdown]
# TODO: complete list of required modules to install

# %%
# %pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
# %pip install youtube-transcript-api TTS soundfile pydub webvtt-py librosa

# %%
import os
import pickle
from youtube_transcript_api import YouTubeTranscriptApi

PICKLED_CAPTIONS = data_folder/'subtitles.pickle'
try:
    with open(PICKLED_CAPTIONS, 'rb') as f:
        temp_captions = pickle.load(f)
except FileNotFoundError:
    temp_captions = YouTubeTranscriptApi.get_transcript(youtube_video_id, languages=['en'])
    print('Loading from PICKLE failed. Downloading from Youtube...')
    with open(PICKLED_CAPTIONS, 'wb') as f:
        pickle.dump(temp_captions, f)

# %%
from typing import List
from dataclasses import dataclass, field

@dataclass
class Caption:
    text: str
    start: float
    duration: float
    def __repr__(self):
            return f"{self.timestamp_start()} --> {self.timestamp_end()}\n{self.text}"
    def __post_init__(self):
            # self.end = self.start + self.duration
            object.__setattr__(self, 'end', self.start + self.duration)  # syntax for case @dataclass(frozen=True)
    def timestamp_start(self) -> str: 
        """ Convert seconds to HH:MM:SS,mmm format """
        hours = int(self.start / 3600)
        minutes = int(self.start / 60 - hours * 60)
        seconds = self.start - hours * 3600 - minutes * 60
        return '{:02d}:{:02d}:{:06.3f}'.format(hours, minutes, seconds)
    def timestamp_end(self) -> str:
        """ Convert seconds to HH:MM:SS,mmm format """
        end = self.start + self.duration
        hours = int(end / 3600)
        minutes = int(end / 60 - hours * 60)
        seconds = end - hours * 3600 - minutes * 60
        return '{:02d}:{:02d}:{:06.3f}'.format(hours, minutes, seconds)

captions: List[Caption] = []
for caption in temp_captions:
    captions.append(Caption(text=caption['text'], start=caption['start'], duration=caption['duration']))


# %% [markdown]
# # Optional cell. Save subtitles on disk in WEBTT and SRT formats.

# %%
import webvtt

vtt = webvtt.WebVTT()
for cap in captions:
    vtt.captions.append(webvtt.Caption(cap.timestamp_start(), cap.timestamp_end(), cap.text))
vtt.save(str(data_folder/youtube_video_id) + '.vtt')
vtt.save_as_srt(str(data_folder/youtube_video_id) + '.srt')

# %% [markdown]
# # Prepare text before processing by NLTK library. Better to have simple sentences and no special chars inside.

# %%
import re

full_text = ' '.join([cap.text for cap in captions])
full_text = full_text.replace('  ', ' ')
full_text = re.sub(r'([A-Za-z]) +(\.)', r"\1\2", full_text)
full_text = re.sub(r'\.+', '.', full_text)
full_text = re.sub(r'\. +\.', '.', full_text)
full_text = full_text.replace('’', "'")
with open(data_folder/'full_text.txt', 'w') as f:  # optional
    f.write(full_text)

# %% [markdown]
# Last chance to make corrections in text or to rewrite.
# You can edit text in full_text.txt and add cell to load changes: 
#
# `
# full_text = open(data_folder/'full_text.txt', 'r').read()
# `
#

# %%
full_text = open(data_folder/'new_full_text.txt', 'r').read()

# %%
from typing import List
from nltk.tokenize import sent_tokenize

sentences: List[str] = sent_tokenize(full_text)


# %% [markdown]
# # Create new_captions: List[NewCaption] from these sentences.
#
# Initial start point and duration is zero (will be set later). 

# %%
from typing import List

@dataclass(frozen=False)
class NewCaption(Caption):
    text: str
    start: float = 0
    duration: float = 0
    wav_path: str = ''
    wav_duration: float = 0
    def meta_info(self) -> str:
        res = ''
        if self.wav_path:
            res = os.path.split(self.wav_path)[1].split('.')[0] + '|' + self.text
        return res

new_captions: List[NewCaption] = []
for sent in sentences:
    new_captions.append(NewCaption(sent))


# %% [markdown]
# # Generate WAV files for each sentence in new_captions

# %% tags=[]
from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer
from tqdm.auto import tqdm, trange
import librosa

manager = ModelManager(tss_models_json_path)
# model_name = "tts_models/en/ljspeech/tacotron2-DCA"
model_name = "tts_models/en/ljspeech/tacotron2-DDC_ph"
# vocoder_name = "vocoder_models/en/ljspeech/multiband-melgan"
vocoder_name = "vocoder_models/en/ljspeech/univnet"
model_path, config_path, model_item = manager.download_model(model_name)
vocoder_path, vocoder_config_path, _ = manager.download_model(vocoder_name)
synthesizer = Synthesizer(
    model_path,
    config_path,
    None,
    None,
    vocoder_path,
    vocoder_config_path,
    None,
    None,
    USE_CUDA,
)

os.makedirs(data_folder/'wavs', exist_ok=True)

resynthesize_existed = False  # for debug purposes

cur = 0
for i, cap in enumerate(tqdm(new_captions)):
    cap.wav_path = str(data_folder/'wavs'/f'{i + 1:03}.wav')
    if resynthesize_existed or not os.path.isfile(cap.wav_path):
        wav = synthesizer.tts(new_captions[i].text, None, None, None, reference_wav=None, reference_speaker_name=None,)
    else:
        wav, _ = librosa.load(cap.wav_path)
    cap.wav_duration = librosa.get_duration(wav)
    cap.duration = cap.wav_duration
    cap.start = cur
    cur = cur + cap.wav_duration
    synthesizer.save_wav(wav, new_captions[i].wav_path)
    print(f'{cap=}\n')

with open(data_folder/'metadata.csv', 'w', encoding='utf-8') as f:
    f.write('\n'.join([cap.meta_info() for cap in new_captions]))

# %% [markdown]
# # Optional. Generate audios (in folder 'wavs-alt') with another model to compare duration of each file to find bad-generated by DCA model with 'attentions'.

# %%
# model_name = "tts_models/en/sam/tacotron-DDC"
# vocoder_name = "vocoder_models/en/sam/hifigan_v2"
model_name = "tts_models/en/ljspeech/tacotron2-DCA"
vocoder_name = "vocoder_models/en/ljspeech/multiband-melgan"

model_path, config_path, model_item = manager.download_model(model_name)
vocoder_path, vocoder_config_path, _ = manager.download_model(vocoder_name)
synthesizer = Synthesizer(
    model_path,
    config_path,
    None,
    None,
    vocoder_path,
    vocoder_config_path,
    None,
    None,
    True,
)
os.makedirs(data_folder/'wavs-alt', exist_ok=True)
for i in range(0, len(new_captions)):
    wav = synthesizer.tts(
        new_captions[i].text,
        None,
        None,
        None,
        reference_wav=None,
        reference_speaker_name=None,
    )
    synthesizer.save_wav(wav, new_captions[i].wav_path.replace('wavs', 'wavs-alt'))


# %% [markdown]
# # Optional. Compare wavs and wavs-alt by duration.

# %%
import soundfile as sf
from pydub import AudioSegment
from tqdm.auto import trange

for i in trange(0, len(new_captions)):
    wav_alt = str(new_captions[i].wav_path).replace('wavs', 'wavs-alt')
    sound_file_alt = sf.SoundFile(wav_alt)
    duration_alt = len(sound_file_alt) / sound_file_alt.samplerate
    ratio = new_captions[i].wav_duration / duration_alt
    filler = ' '
    if ratio < 0.97:
        print(f'{i + 1:03}.wav {new_captions[i].wav_duration:{filler}>7.3f}  ~~>{duration_alt:{filler}>7.3f}         Ratio:{ratio:{filler}>7.2f}')


# %% [markdown]
# # Optional. Process audio files by DeepSpeech.

# %%
# pip install numpy deepspeech deepspeech-gpu

import wave
from deepspeech import Model 
import numpy as np
import librosa
from tqdm.autonotebook import tqdm


ds_model = 'S:/temp/data/models/deepspeech-0.9.3-models.pbmm'
ds = Model(ds_model)
ds_scorer = 'S:/temp/data/models/deepspeech-0.9.3-models.scorer'
ds.enableExternalScorer(ds_scorer)

recognized_texts: List[str] = []
for cap in tqdm(new_captions):
    y, sr = librosa.load(cap.wav_path, sr=16000)
    audio = (y * 32767).astype(np.int16)
    recognized_texts.append(ds.stt(audio))

# %% [markdown]
# # Optional. Compare recognized texts with original text in subtitles.

# %%
for i, sent in enumerate(recognized_texts):
    num_words_in_res = len(sent.split())
    num_words_in_original = len(new_captions[i].text.split())
    ratio = num_words_in_res / num_words_in_original
    if num_words_in_original < 10:
      continue
    if ratio <= 0.85:
        print(f'\n!!! {i + 1:03}.wav ---------------------------------------------------------- ratio = {ratio:3.2f}')
        print(sent)
        print(new_captions[i].text)


# %% [markdown]
# # Optional. Visualize beginning state of subtitles. All wavs aligned to left on timeline.

# %%
# %matplotlib qt
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
whole_video_duration_by_captions = int(captions[-1].start + captions[-1].duration) + 1
cols = 60
rows = whole_video_duration_by_captions // 60 + 1
rounder_dur = whole_video_duration_by_captions + (60 - whole_video_duration_by_captions % 60)

time_matrix = np.zeros(rounder_dur, dtype=np.int)
for i, cap in enumerate(new_captions):
    time_matrix[int(cap.start)] = i + 1
    for j in range(1, math.ceil(cap.duration)):
        time_matrix[int(cap.start) + j] = i + 1
time_matrix = time_matrix.reshape(rows, 60)

time_matrix_without_zeros = time_matrix.copy()
time_matrix_without_zeros = time_matrix_without_zeros.astype(str)
time_matrix_without_zeros[time_matrix_without_zeros == '0'] = ''

# %matplotlib qt
num_fo_captions = len(new_captions)
palette = [(0, 0, 0)]
colors = [(0.860, 0.371, 0.339), (0.568, 0.860, 0.339), (0.631, 0.400, 0.860)]
colors *= num_fo_captions // len(colors) + 1
palette.extend(colors)
new_cmap = matplotlib.colors.ListedColormap(palette)
norm = matplotlib.colors.BoundaryNorm(np.arange(0, num_fo_captions + 1), num_fo_captions)
ax = sns.heatmap(time_matrix, linewidth=0, annot=time_matrix_without_zeros, fmt="s", cbar=None, cmap=new_cmap, norm=norm)



# %% [markdown]
# # Creating new **arranged_new_captions** list and arranging **start** for each WAV file to correct start point on timeline by matching text with text in CAPTIONS.

# %%
import time
import math
import string
import copy


arranged_new_captions = copy.deepcopy(new_captions)
whole_video_duration_by_captions = int(captions[-1].start + captions[-1].duration) + 1
end_point = whole_video_duration_by_captions
indexes: List[int] = []
indexes.append(len(captions) - 1)  # first index = last

show_debug = False
def debug(x):
    if show_debug:
        print(x)

debug(f'Number of sentences in new captions: {len(new_captions)}')
debug(f'Number of sentences in original captions: {len(captions)}')
# Starts moving from last to first subtitle.
# Tries to find correct start time for each by comparing text with original (not reshaped to sentences) subtitles. 
# If can not find correspondence, makes it 'float' (can be moved by neighbor)
for new_cap in reversed(arranged_new_captions): 
    debug(indexes)
    debug(f'\n\n{new_cap=}')
    congruence_index = len(captions) - 1  # default for case if no correspondent test found
    # Squeezes sentence to string only from ascii letters. So search will work more reliable.
    # Example: This is a text. --> Thisisatext
    sub = ''.join(filter(lambda x: x in string.ascii_letters , new_cap.text))
    cap_block = ''  # caption block from captions (squeezed by the same way)
    
    for i, cap in reversed(list(enumerate(captions))):
        # Accumulates captions in cap_block until sentence is found
        cap_block = ''.join(filter(lambda x: x in string.ascii_letters , cap.text)) + cap_block
        if sub in cap_block:
            # Every found subtitle index must be equal or less then previous.
            if len(indexes) > 0 and i <= indexes[-1]:
                congruence_index = i
                break
            else:
                # If not - remove sub from cap_block to search sub in next subtitles
                cap_block = cap_block.replace(sub, '')  # Skip this. We will look in next block left
    if congruence_index == len(captions) - 1:
        debug('No correspondent block found.')
        congruence_index = indexes[-1]
    indexes.append(congruence_index)
  
    start = captions[congruence_index].start
    new_cap.start = min(start, (end_point - new_cap.duration)) if new_cap.start < start else new_cap.start
    end_point = new_cap.start


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

discrete_difference = np.diff(np.array(indexes))
if not all(discrete_difference <= 0):
    print('Descending sequence check failed. Something went wrong with indexes.')

cols = 60
rows = whole_video_duration_by_captions // 60 + 1
# Round DURATION in seconds up to the next multiple of 60
rounder_dur = whole_video_duration_by_captions + (60 - whole_video_duration_by_captions % 60)

time_matrix_2 = np.zeros(rounder_dur,  dtype=np.int)
for i, cap in enumerate(arranged_new_captions):
    time_matrix_2[int(cap.start)] = i + 1
    for j in range(1, math.ceil(cap.duration)):
        time_matrix_2[int(cap.start) + j] = i + 1

time_matrix_2 = time_matrix_2.reshape(rows, 60)
time_matrix_2_without_zeros = time_matrix_2.copy()
time_matrix_2_without_zeros = time_matrix_2_without_zeros.astype(str)
time_matrix_2_without_zeros[time_matrix_2_without_zeros == '0'] = ''


# %matplotlib qt
num_fo_captions = len(new_captions)
palette = [(0, 0, 0)]
colors = [(0.860, 0.371, 0.339), (0.568, 0.860, 0.339), (0.631, 0.400, 0.860)]
colors *= num_fo_captions // len(colors) + 1
palette.extend(colors)
new_cmap = matplotlib.colors.ListedColormap(palette)
norm = matplotlib.colors.BoundaryNorm(np.arange(0, num_fo_captions + 1), num_fo_captions)
ax = sns.heatmap(time_matrix_2, linewidth=0, annot=time_matrix_2_without_zeros, fmt="s", cbar=None, cmap=new_cmap, norm=norm)


# %% [markdown]
# # Concatenate wavs in 'final.wav'

# %%
from pydub import AudioSegment

audio_track = AudioSegment.silent(duration=whole_video_duration_by_captions * 1000)
for cap in tqdm(arranged_new_captions):
    sound = AudioSegment.from_file(cap.wav_path, format="wav")
    audio_track = audio_track.overlay(sound, position=cap.start * 1000)
audio_track.export(data_folder/'wavs'/'final.wav', format="wav").close()

# %%
# # Useless caption processing version partially in func-style with map/reduce.

# import itertools
# import functools
# import math
# import string
# import copy

# arranged_new_captions = copy.deepcopy(new_captions)

# def shrink(text: str) -> str:
#     return ''.join(filter(lambda c: c in string.ascii_letters , text))
    
# def reduce_search(x, y):
#     if x[2] in x[1]:
#         if x[0] <= x[3]:
#             return x
#         else:
#             x[1] = x[1].replace(x[2], '')
#     return  (y[0], y[1] + x[1], x[2], y[3])

# whole_video_duration_by_captions = int(captions[-1].start + captions[-1].duration) + 1
# end_point = whole_video_duration_by_captions
# indexes: List[int] = []
# indexes.append(len(captions) - 1)  # first index = last

# captions_for_reduce = list(reversed(list(enumerate(captions))))
# captions_for_reduce = list(itertools.starmap(lambda x, cap: (x, shrink(cap.text)), captions_for_reduce))

# for new_cap in reversed(arranged_new_captions): 
#     data = list(itertools.starmap(
#         lambda x, captext: (x, captext, shrink(new_cap.text), indexes[-1]),
#         captions_for_reduce))
#     congruence_index = list(functools.reduce(reduce_search, data))[0]

#     if not congruence_index:
#         congruence_index = len(captions) - 1
#     indexes.append(congruence_index)
  
#     start = captions[congruence_index].start
#     new_cap.start = min(start, (end_point - new_cap.duration)) if new_cap.start < start else new_cap.start
#     end_point = new_cap.start
