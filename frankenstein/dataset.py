import numpy as np
import torch
from torch.utils.data import Dataset
import json
import random
from pathlib import Path
import os


class CPCDataset_sameSeq(Dataset):
    def __init__(self, root, n_sample_frames, mode):
        self.root = Path(root)
        self.n_sample_frames = n_sample_frames

        if mode == "test" or mode == "plot" or mode == "plot_spk" or mode == "plots":
            # For test mode, look inside 'root/test/Angry/mels'
            if 'esd' in str(root).lower():
                self.speakers = sorted(os.listdir(root/f'test/Angry/mels'))
            else:
                self.speakers = sorted(os.listdir(root/f'test/ANG/mels'))
                        
        else:
            # For other modes, original behavior
            self.speakers = sorted(os.listdir(root/f'{mode}/mels'))

        
        with open(self.root / f"{mode}.json") as file:
            metadata = json.load(file)
        self.metadata = []
        for mel_len, mel_out_path, lf0_out_path in metadata:
            # if mel_len > n_sample_frames: # only select wavs having frames>=140
            mel_out_path = Path(mel_out_path)
            lf0_out_path = Path(lf0_out_path)
            speaker = mel_out_path.parent.stem
            self.metadata.append([speaker, mel_out_path, lf0_out_path])
        print('n_sample_frames:', n_sample_frames, 'metadata:', len(self.metadata))
        random.shuffle(self.metadata)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        speaker, mel_path, lf0_path = self.metadata[index]

        mel_path = self.root.parent / mel_path
        lf0_path = self.root.parent / lf0_path
        mel = np.load(mel_path).T
        lf0 = np.load(lf0_path)
        melt = mel
        lf0t = lf0
        while mel.shape[-1] < self.n_sample_frames:
            mel = np.concatenate([mel, melt], -1)
            lf0 = np.concatenate([lf0, lf0t], 0)
        zero_idxs = np.where(lf0 == 0.0)[0]
        nonzero_idxs = np.where(lf0 != 0.0)[0]
        if len(nonzero_idxs) > 0 :
            mean = np.mean(lf0[nonzero_idxs])
            std = np.std(lf0[nonzero_idxs])
            if std == 0:
                lf0 -= mean
                lf0[zero_idxs] = 0.0
            else:
                lf0 = (lf0 - mean) / (std + 1e-8)
                lf0[zero_idxs] = 0.0

        pos = random.randint(0, mel.shape[-1] - self.n_sample_frames)
        mel = mel[:, pos:pos + self.n_sample_frames]
        lf0 = lf0[pos:pos + self.n_sample_frames]

        # ======= Emotion label extraction ======= #
        label_str = -1
        mel_path_str = str(mel_path)

        if 'esd' in mel_path_str.lower():
            # e.g., 0011_000352.npy => index=352
            try:
                idx = int(mel_path.stem.split('_')[1])
            except (IndexError, ValueError):
                idx = 0
            if 1 <= idx <= 350:
                label_str = 'NEU'
            elif 351 <= idx <= 700:
                label_str = 'ANG'
            elif 701 <= idx <= 1050:
                label_str = 'HAP'
            elif 1051 <= idx <= 1400:
                label_str = 'SAD'
            elif 1401 <= idx <= 1750:
                label_str = 'SUR'
        else:
            label_str = mel_path.stem.split('_')[2]

        if 'esd' in mel_path_str.lower():
            EMOTION_MAP = {'NEU': 0, 'ANG': 1, 'HAP': 2, 'SAD': 3, 'SUR': 4}
        else:
            EMOTION_MAP = {'NEU': 0, 'ANG': 1, 'HAP': 2, 'SAD': 3, 'FEA': 4, 'DIS': 5}

        label = EMOTION_MAP[label_str]
        # print(f"Extracted emotion label: {label_str} ({label}) from {mel_path_str}")
        # print(self.speakers.index(speaker))

        return label, torch.from_numpy(mel), torch.from_numpy(lf0), self.speakers.index(speaker)




