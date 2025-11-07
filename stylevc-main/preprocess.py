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
dataset_used = 'CREMAD'
if (dataset_used == 'ESD'):
    data_root = '../ESD_Reorganized'
    emotions = ["Angry", "Happy", "Neutral", "Sad", "Surprise"]
    save_root = './data_esd'
    all_spks = os.listdir(data_root)
    train_spks = all_spks[:8]
    test_spks = all_spks[8:10]
    num_val = 30
else :
    data_root = '../CREMA-D_Reorganized'
    emotions = ["ANG", "DIS", "FEA", "HAP", "NEU", "SAD"]
    save_root = './data_cremad'
    all_spks = os.listdir(data_root)
    train_spks = all_spks[:82]
    test_spks = all_spks[82:91]
    num_val = 2

# data_root = '../ESD_Reorganized'
# save_root = './data_esd'
# emotion = "Angry"
# emotions = ["Angry", "Happy", "Neutral", "Sad", "Surprise"]
# data_root = '../CREMA-D_Reorganized'
# emotions = ["ANG", "DIS", "FEA", "HAP", "NEU", "SAD"]
# save_root = './data_cremad'
# all_spks = os.listdir(data_root)
# print('all_spks:', all_spks)
# train_spks = all_spks[:82]
# test_spks = all_spks[82:91]


os.makedirs(save_root, exist_ok=True)



all_spks = os.listdir(data_root)
# train_spks = all_spks[:8]
# test_spks = all_spks[8:10]

train_wavs_names = []
valid_wavs_names = []
test_wavs_names_by_emotion = {e: [] for e in emotions}

# Collect data
for emotion in emotions:
    for spk in train_spks:
        spk_wavs = glob(f'{data_root}/{spk}/{emotion}/train/*.wav')
        spk_wavs_names = [os.path.basename(p).split('.')[0] for p in spk_wavs]
        valid_names = random.sample(spk_wavs_names, min(2, len(spk_wavs_names)))
        train_names = [n for n in spk_wavs_names if n not in valid_names]
        
        train_wavs_names += train_names
        valid_wavs_names += valid_names

        test_paths = glob(f'{data_root}/{spk}/{emotion}/test/*.wav')
        test_names = [os.path.basename(p).split('.')[0] for p in test_paths]
        test_wavs_names_by_emotion[emotion] += test_names

    for spk in test_spks:
        spk_wavs = glob(f'{data_root}/{spk}/{emotion}/test/*.wav')
        spk_wavs_names = [os.path.basename(p).split('.')[0] for p in spk_wavs]
        test_wavs_names_by_emotion[emotion] += spk_wavs_names

# Extract features from all wavs
all_wavs = glob(f'{data_root}/*/*/*/*.wav')
all_wavs = [item.replace('\\', '/') for item in all_wavs]
results = Parallel(n_jobs=-1)(delayed(extract_logmel)(wav_path) for wav_path in tqdm(all_wavs))
wn2mel = {r[0]: [r[1], r[2], r[3]] for r in results}

# Normalize
mels = [wn2mel[wav_name][0] for wav_name in train_wavs_names if wav_name in wn2mel]
mels = np.concatenate(mels, axis=0)
mean, std = np.mean(mels, axis=0), np.std(mels, axis=0)
np.save(f'{save_root}/mel_stats.npy', np.stack([mean, std]))

results = Parallel(n_jobs=-1)(delayed(normalize_logmel)(wav_name, wn2mel[wav_name][0], mean, std) for wav_name in tqdm(wn2mel.keys()))
wn2mel_new = {wav_name: [mel, wn2mel[wav_name][1], wn2mel[wav_name][2]] for wav_name, mel in results}

# Save train/valid (emotion-agnostic)
train_results = Parallel(n_jobs=-1)(delayed(save_logmel)(
    save_root, wav_name, wn2mel_new[wav_name], 'train') for wav_name in tqdm(train_wavs_names))

valid_results = Parallel(n_jobs=-1)(delayed(save_logmel)(
    save_root, wav_name, wn2mel_new[wav_name], 'valid') for wav_name in tqdm(valid_wavs_names))

# Save test (split by emotion)
test_results = []
for emotion in emotions:
    res = Parallel(n_jobs=-1)(delayed(save_logmel)(
        save_root, wav_name, wn2mel_new[wav_name], 'test', emotion) for wav_name in tqdm(test_wavs_names_by_emotion[emotion], desc=f"Test: {emotion}"))
    test_results += res

save_json(save_root, train_results, 'train')
save_json(save_root, valid_results, 'valid')
save_json(save_root, test_results, 'test')  # or individual test_emotion.jsons if preferred


    


