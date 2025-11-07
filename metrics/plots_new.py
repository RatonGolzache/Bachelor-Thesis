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


# EMOTION_MAP = {'NEU': 0, 'ANG': 1, 'HAP': 2, 'SAD': 3, 'SUR': 4}
EMOTION_MAP = {'NEU': 0, 'ANG': 1, 'HAP': 2, 'SAD': 3, 'FEA': 4, 'DIS': 5}
EMOTION_MAP_INV = {v: k for k, v in EMOTION_MAP.items()}

def plot_with_labels(embeddings, labels, speakers, method='pca', title='', save_path=None, mode='spk'):

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
        emotion = EMOTION_MAP_INV.get(lbl, str(lbl))
        plt.scatter(reduced[idx, 0], reduced[idx, 1], label=emotion, alpha=0.6)
    plt.legend()
    plt.title(f"{title} ({method.upper()}) colored by emotion")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True)
    if save_path is not None:
        plt.savefig(save_path  / (mode + '_tsne_cbe.png'))
        plt.close()  # Close the plot after saving to free memory
    else:
        plt.show()


    plt.figure(figsize=(8,6))
    for spk in np.unique(speakers):
        idx = speakers == spk
        plt.scatter(reduced[idx, 0], reduced[idx, 1], label=str(spk), alpha=0.6)
    plt.legend()
    plt.title(f"{title} ({method.upper()}) colored by speaker")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True)
    if save_path is not None:
        plt.savefig(save_path / (mode + '_tsne_cbs.png'))
        plt.close()  # Close the plot after saving to free memory
    else:
        plt.show()

    


@hydra.main(config_path="./config/train.yaml")
def train_model(cfg):
    
    #cfg.checkpoint_dir = f'{cfg.checkpoint_dir}'
    if cfg.encoder_lf0_type == 'no_emb':  # default
        dim_lf0 = 1
    else:
        dim_lf0 = 64

    result_dir = Path(utils.to_absolute_path("plots_gcl1_cls_cremad"))
    result_dir.mkdir(exist_ok=True, parents=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # define model
    encoder = Encoder(**cfg.model.encoder) # config -> model -> default.yaml
    encoder_lf0 = Encoder_lf0(cfg.encoder_lf0_type) # no_emb
    encoder_spk = Encoder_spk() # speaker encoder
    encoder_emo = EmoEncoder() # style encoder

    encoder.to(device)
    encoder_lf0.to(device)
    encoder_spk.to(device)
    encoder_emo.to(device)

    # Load dataset
    root_path = Path(utils.to_absolute_path(cfg.data_root))
    print(root_path)
    dataset = CPCDataset(
        root=root_path,
        n_sample_frames=cfg.training.sample_frames,  # 128
        mode='plots')

    # Load data
    dataloader = DataLoader(
        dataset,
        batch_size=len(dataset), 
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False)


    # Load checkpoint
    print("Load encoder checkpoint from: {}:".format(cfg.path_for_plots))
    model_path = utils.to_absolute_path(cfg.path_for_plots)
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    encoder.load_state_dict(checkpoint["encoder"])
    encoder_lf0.load_state_dict(checkpoint["encoder_lf0"])
    encoder_spk.load_state_dict(checkpoint["encoder_spk"])
    encoder_emo.load_state_dict(checkpoint["encoder_emo"])
    
    for i, (labels, mels, lf0, speakers) in enumerate(dataloader, 1):
        lf0 = lf0.to(device)
        mels = mels.to(device) 
        labels = labels.to(device)  
        speakers = speakers.to(device)  

    z, c, _, vq_loss, perplexity = encoder(mels)
    lf0_embs = encoder_lf0(lf0)

    spk_embs= encoder_spk(mels)
    emo_embs = encoder_emo(mels)

    # Upsample z to match the mel-spectrogram length
    z = F.interpolate(z.transpose(1, 2), scale_factor=2) # (bs, 140/2, 64) -> (bs, 64, 140/2) -> (bs, 64, 140)
    z = z.transpose(1, 2) # (bs, 64, 140) -> (bs, 140, 64)
    # Upsample speaker embedding to match the mel-spectrogram length by repeating it
    spk_embs_exp = spk_embs.unsqueeze(1).expand(-1,z.shape[1],-1)
    lf0_embs = lf0_embs[:,:z.shape[1],:]

    # Upsample emotion embedding to match the mel-spectrogram length by repeating it
    emo_embs_exp = emo_embs.unsqueeze(1).expand(-1,z.shape[1],-1)
    # print(z.shape, lf0_embs.shape)
    # Concatenate the embeddings and the fundamental frequency
    x = torch.cat([z, lf0_embs, spk_embs_exp, emo_embs_exp], dim=-1)
    print(x.shape)
    print(labels.shape, speakers.shape)

    # Convert tensors to numpy and flatten for PCA/t-SNE
    speakers_np = speakers.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    x_np = x.detach().cpu().numpy().reshape(x.shape[0], -1)
    z_np = z.detach().cpu().numpy().reshape(z.shape[0], -1)
    spk_np = spk_embs.detach().cpu().numpy().reshape(spk_embs_exp.shape[0], -1)
    lf0_np = lf0_embs.detach().cpu().numpy().reshape(lf0_embs.shape[0], -1)
    emo_np = emo_embs.detach().cpu().numpy().reshape(emo_embs_exp.shape[0], -1)

    # Save plots
    plot_with_labels(spk_np, labels_np, speakers_np, method='tsne', title='Speaker Encoder', save_path=result_dir, mode = 'spk')
    plot_with_labels(emo_np, labels_np, speakers_np, method='tsne', title='Emotion Encoder', save_path=result_dir, mode = 'emo')


if __name__ == "__main__":
    train_model()
