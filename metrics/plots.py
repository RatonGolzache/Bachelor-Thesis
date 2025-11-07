import hydra
from hydra import utils
import kaldiio
import numpy as np
from pathlib import Path
import subprocess
from itertools import chain

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from dataset import CPCDataset_sameSeq as CPCDataset
from model_encoder import Encoder, CPCLoss_sameSeq, Encoder_lf0, VAEncoder
from model_encoder import EmoEncoder
from model_encoder import SpeakerEncoder as Encoder_spk

import os
import time

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

torch.cuda.empty_cache()
torch.manual_seed(137)
np.random.seed(137)


from pathlib import Path

#EMOTION_MAP = {'NEU': 0, 'ANG': 1, 'HAP': 2, 'SAD': 3, 'SUR': 4}
EMOTION_MAP = {'NEU': 0, 'ANG': 1, 'HAP': 2, 'SAD': 3, 'FEA': 4, 'DIS': 5}
EMOTION_MAP_INV = {v: k for k, v in EMOTION_MAP.items()}

def plot_with_labels_no_legend(embeddings, labels, speakers, method='pca', title='', save_path=None, mode='spk'):
    if method == 'pca':
        reducer = PCA(n_components=2)
    elif method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=29)
    else:
        raise ValueError("Method must be 'pca' or 'tsne'")

    reduced = reducer.fit_transform(embeddings)
    plt.figure(figsize=(8,6))
    for lbl in np.unique(labels):
        idx = labels == lbl
        plt.scatter(reduced[idx, 0], reduced[idx, 1], alpha=0.6, s=40)
    #plt.title(f"{title} colored by Emotion",  fontsize=18)
    plt.xticks([])
    plt.yticks([])
    if save_path is not None:
        plt.savefig(save_path / (mode + '_tsne_cbe.png'))
        plt.close()  # Close after saving to free memory
    else:
        plt.show()
    if save_path is not None:
        legend_path = save_path / (mode + '_color_key.txt')
        with open(legend_path, 'w') as f:
            if mode == 'spk':
                for spk in np.unique(speakers):
                    f.write(f"Speaker ID: {spk}\n")
            else:
                for lbl in np.unique(labels):
                    emotion = EMOTION_MAP_INV.get(lbl, str(lbl))
                    f.write(f"Emotion label {lbl}: {emotion}\n")

    plt.figure(figsize=(8,6))
    for spk in np.unique(speakers):
        idx = speakers == spk
        plt.scatter(reduced[idx, 0], reduced[idx, 1], alpha=0.6, s=40)
    #plt.title(f"{title} colored by Speaker",  fontsize=18)
    plt.xticks([])
    plt.yticks([])
    if save_path is not None:
        plt.savefig(save_path / (mode + '_tsne_cbs.png'))
        plt.close()
    else:
        plt.show()
    if save_path is not None:
        legend_path = save_path / (mode + '_color_key.txt')
        with open(legend_path, 'w') as f:
            if mode == 'spk':
                for spk in np.unique(speakers):
                    f.write(f"Speaker ID: {spk}\n")
            else:
                for lbl in np.unique(labels):
                    emotion = EMOTION_MAP_INV.get(lbl, str(lbl))
                    f.write(f"Emotion label {lbl}: {emotion}\n")


@hydra.main(config_path="./config/metrics.yaml")
def train_model(cfg):
    # Create a base plots directory
    base_plot_dir = Path(utils.to_absolute_path("plots"))
    base_plot_dir.mkdir(exist_ok=True, parents=True)


    dim_lf0 = 1


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = Encoder(**cfg.model.encoder)
    encoder_lf0 = Encoder_lf0('no_emb')
    encoder_spk = Encoder_spk()
    encoder_emo = EmoEncoder()

    encoder.to(device)
    encoder_lf0.to(device)
    encoder_spk.to(device)
    encoder_emo.to(device)

    root_path = Path(utils.to_absolute_path(cfg.data_root_plots))
    dataset = CPCDataset(root=root_path, n_sample_frames=cfg.training.sample_frames, mode='plots')
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=0, pin_memory=True, drop_last=False)

    model_folder = Path(utils.to_absolute_path(cfg.checkpoint_dir))

    pt_files = list(model_folder.glob("*.pt"))

    if not pt_files:
        print(f"No .pt files found in the directory {model_folder}")
        return

    for model_path in pt_files:
        print(f"Loading encoder checkpoint from: {model_path}")
        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
        encoder.load_state_dict(checkpoint["encoder"])
        encoder_lf0.load_state_dict(checkpoint["encoder_lf0"])
        encoder_spk.load_state_dict(checkpoint["encoder_spk"])
        encoder_emo.load_state_dict(checkpoint["encoder_emo"])

        # Create subfolder named after the model file (without extension)
        model_name = model_path.stem
        result_dir = base_plot_dir / model_name
        result_dir.mkdir(exist_ok=True, parents=True)

        for i, (labels, mels, lf0, speakers) in enumerate(dataloader, 1):
            lf0 = lf0.to(device)
            mels = mels.to(device)
            labels = labels.to(device)
            speakers = speakers.to(device)

        z, c, _, vq_loss, perplexity = encoder(mels)
        lf0_embs = encoder_lf0(lf0)
        spk_embs = encoder_spk(mels)
        emo_embs = encoder_emo(mels)

        z = F.interpolate(z.transpose(1, 2), scale_factor=2)
        z = z.transpose(1, 2)
        spk_embs_exp = spk_embs.unsqueeze(1).expand(-1, z.shape[1], -1)
        lf0_embs = lf0_embs[:, :z.shape[1], :]
        emo_embs_exp = emo_embs.unsqueeze(1).expand(-1, z.shape[1], -1)

        x = torch.cat([z, lf0_embs, spk_embs_exp, emo_embs_exp], dim=-1)

        speakers_np = speakers.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        z_np = z.detach().cpu().numpy().reshape(z.shape[0], -1)
        spk_np = spk_embs.detach().cpu().numpy().reshape(spk_embs_exp.shape[0], -1)
        lf0_np = lf0_embs.detach().cpu().numpy().reshape(lf0_embs.shape[0], -1)
        emo_np = emo_embs.detach().cpu().numpy().reshape(emo_embs_exp.shape[0], -1)

        # Save plots without legends
        plot_with_labels_no_legend(spk_np, labels_np, speakers_np, method='tsne', title='Speaker Encoder', save_path=result_dir, mode='spk')
        plot_with_labels_no_legend(emo_np, labels_np, speakers_np, method='tsne', title='Emotion Encoder', save_path=result_dir, mode='emo')
        plot_with_labels_no_legend(z_np, labels_np, speakers_np, method='tsne', title='Content Encoder', save_path=result_dir, mode='content')


if __name__ == "__main__":
    train_model()
