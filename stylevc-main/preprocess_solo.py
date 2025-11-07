# -*- coding: utf-8 -*-

from spectrogram import logmelspectrogram
import numpy as np
from joblib import Parallel, delayed
import librosa
import soundfile as sf
import os
from glob import glob
from tqdm import tqdm
import random
import json
import resampy
import pyworld as pw


# Function to extract a logmel spectrogram and lf0 from a wav file using pyworld and librosa. 
# The function takes the path to the wav file and the sample rate as input.
# It loads the wav file, trims silence, and resamples it to the desired sample rate.
# It then computes the logmel spectrogram and f0 using pyworld.
def extract_logmel(wav_path, sr=16000):
    # wav, fs = librosa.load(wav_path, sr=sr)
    wav, fs = sf.read(wav_path)
    wav, _ = librosa.effects.trim(wav, top_db=60)
    if fs != sr:
        wav = resampy.resample(wav, fs, sr, axis=0)
        fs = sr
    # duration = len(wav)/fs
    assert fs == 16000
    peak = np.abs(wav).max()
    if peak > 1.0:
        wav /= peak
    # n_fft is the number of samples in the FFT window, n_shift is the number of samples to shift the window
    # for each frame and win_length is the length of the window. The window function is set to 'hann', 
    # and the minimum and maximum frequencies are set to 80 Hz and 7600 Hz, respectively.
    mel_spectrogram = logmelspectrogram(
                x=wav,
                fs=fs,
                n_mels=80,
                n_fft=400,
                n_shift=160,
                win_length=400,
                window='hann',
                fmin=80,
                fmax=7600,
            )
    
    # tlen is the length of the mel spectrogram, and frame_period is the time between frames in milliseconds.
    tlen = mel_spectrogram.shape[0]
    frame_period = 160/fs*1000

    # Compute the f0 using pyworld. The dio function estimates the f0 using the DIO algorithm, 
    # and the stonemask function refines the f0 estimate.
    f0, timeaxis = pw.dio(wav.astype('float64'), fs, frame_period=frame_period)
    f0 = pw.stonemask(wav.astype('float64'), f0, timeaxis, fs)
    f0 = f0[:tlen].reshape(-1).astype('float32')
    nonzeros_indices = np.nonzero(f0)
    lf0 = f0.copy()
    lf0[nonzeros_indices] = np.log(f0[nonzeros_indices]) # for f0(Hz), lf0 > 0 when f0 != 0
    
    wav_name = os.path.basename(wav_path).split('.')[0]
    # print(wav_name, mel.shape, duration)
    # return name, mel, lf0, duration
    return wav_name, mel_spectrogram, lf0, mel_spectrogram.shape[0]


def normalize_logmel(wav_name, mel_spectrogram, mean, std):
    mel_spectrogram = (mel_spectrogram - mean) / (std + 1e-8)
    return wav_name, mel_spectrogram


def save_one_file(save_path, arr):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, arr)


def save_logmel(save_root, wav_name, melinfo, mode, emotion=None):
    spk = wav_name.split('_')[0]
    if mode == 'test' and emotion is not None:
        mel_save_path = f'{save_root}/{mode}/{emotion}/mels/{spk}/{wav_name}.npy'
        lf0_save_path = f'{save_root}/{mode}/{emotion}/lf0/{spk}/{wav_name}.npy'
    else:
        mel_save_path = f'{save_root}/{mode}/mels/{spk}/{wav_name}.npy'
        lf0_save_path = f'{save_root}/{mode}/lf0/{spk}/{wav_name}.npy'

    mel_spectrogram, lf0, mel_len = melinfo
    save_one_file(mel_save_path, mel_spectrogram)
    save_one_file(lf0_save_path, lf0)
    return mel_len, mel_save_path, lf0_save_path

def save_json(save_root, results, mode):
    fp = open(f'{save_root}/{mode}.json', 'w')
    json.dump(results, fp, indent=4)
    fp.close()
    


#############################################################################################################
data_root = '../ESD_Reorganized'
save_root = './data_esd_neutral'
emotion = "Neutral"
os.makedirs(save_root, exist_ok=True)

all_spks = os.listdir(data_root)
train_spks = all_spks[:8]
test_spks = all_spks[8:10]

train_wavs_names = []
valid_wavs_names = []
test_wavs_names = []

print('all_spks:', all_spks)
for spk in train_spks:
    spk_wavs = glob(f'{data_root}/{spk}/{emotion}/train/*.wav')

    print('len(spk_wavs):', len(spk_wavs))
    spk_wavs_names = [os.path.basename(p).split('.')[0] for p in spk_wavs]
    valid_names = random.sample(spk_wavs_names, 30)
    n_l = [n for n in spk_wavs_names if n not in valid_names]
    test_names = glob(f'{data_root}/{spk}/{emotion}/test/*.wav')
    test_names_new = [os.path.basename(p).split('.')[0] for p in test_names]
    new_list = valid_names + test_names
    train_names = [n for n in spk_wavs_names if n not in new_list]
    train_wavs_names += train_names
    valid_wavs_names += valid_names
    test_wavs_names += test_names_new


for spk in test_spks:
    spk_wavs = glob(f'{data_root}/{spk}/{emotion}/test/*.wav')
    print('len(spk_wavs):', len(spk_wavs))
    spk_wavs_names = [os.path.basename(p).split('.')[0] for p in spk_wavs]
    test_wavs_names += spk_wavs_names
    
print(len(train_wavs_names))
print(len(valid_wavs_names))
print(len(test_wavs_names))
# extract log-mel
print('extract log-mel...')
all_wavs = glob(f'{data_root}/*/{emotion}/*/*.wav')
all_wavs_new = [item.replace('\\','/') for item in all_wavs]
all_wavs = all_wavs_new

results = Parallel(n_jobs=-1)(delayed(extract_logmel)(wav_path) for wav_path in tqdm(all_wavs))
wn2mel = {}
for r in results:
    wav_name, mel, lf0, mel_len = r
    # print(wav_name, mel.shape, duration)
    wn2mel[wav_name] = [mel, lf0, mel_len]

# normalize log-mel
print('normalize log-mel...')
mels = []
spk2lf0 = {}
for wav_name in train_wavs_names:
    mel, _, _ = wn2mel[wav_name]
    mels.append(mel)

mels = np.concatenate(mels, 0)
mean = np.mean(mels, 0)
std = np.std(mels, 0)
mel_stats = np.concatenate([mean.reshape(1,-1), std.reshape(1,-1)], 0)
np.save(f'{save_root}/mel_stats.npy', mel_stats)

results = Parallel(n_jobs=-1)(delayed(normalize_logmel)(wav_name, wn2mel[wav_name][0], mean, std) for wav_name in tqdm(wn2mel.keys()))
wn2mel_new = {}
for r in results:
    wav_name, mel = r
    lf0 = wn2mel[wav_name][1]
    mel_len = wn2mel[wav_name][2]
    wn2mel_new[wav_name] = [mel, lf0, mel_len]

# save log-mel
print('save log-mel...')
train_results = Parallel(n_jobs=-1)(delayed(save_logmel)(save_root, wav_name, wn2mel_new[wav_name], 'train') for wav_name in tqdm(train_wavs_names))
valid_results = Parallel(n_jobs=-1)(delayed(save_logmel)(save_root, wav_name, wn2mel_new[wav_name], 'valid') for wav_name in tqdm(valid_wavs_names))
test_results = Parallel(n_jobs=-1)(delayed(save_logmel)(save_root, wav_name, wn2mel_new[wav_name], 'test') for wav_name in tqdm(test_wavs_names))

save_json(save_root, train_results, 'train')
save_json(save_root, valid_results, 'valid')
save_json(save_root, test_results, 'test')


    


