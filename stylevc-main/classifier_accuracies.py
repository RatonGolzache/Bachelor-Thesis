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

from dataset import CPCDataset_sameSeq as CPCDataset
from model_encoder import Encoder, CPCLoss_sameSeq, Encoder_lf0
from model_encoder import EmoEncoder
from model_encoder import SpeakerEncoder as Encoder_spk
from model_encoder import ClassifierSpk, ClassifierContent, EmotionClassifier, ClassifierEmo

import os
import time

torch.cuda.empty_cache()
torch.manual_seed(137)
np.random.seed(137)


@hydra.main(config_path="./config/train.yaml")
def train_model(cfg):
    
    #cfg.checkpoint_dir = f'{cfg.checkpoint_dir}'
    if cfg.encoder_lf0_type == 'no_emb':  # default
        dim_lf0 = 1
    else:
        dim_lf0 = 64

    result_dir = Path(utils.to_absolute_path("accuracies"))
    result_dir.mkdir(exist_ok=True, parents=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # define model
    encoder = Encoder(**cfg.model.encoder) # config -> model -> default.yaml
    encoder_lf0 = Encoder_lf0(cfg.encoder_lf0_type) # no_emb
    encoder_spk = Encoder_spk() # speaker encoder
    encoder_emo = EmoEncoder() # style encoder

    classifier_spk = ClassifierSpk(256, 128, cfg.classifier_num_speakers) # classifier from speaker embedding
    classifier_emo = ClassifierEmo(256, 128, cfg.classifier_num_speakers) # classifier from emotion embedding
    classifier_content = ClassifierContent(64, 32, cfg.classifier_num_speakers) # classifier from content embedding

    encoder.to(device)
    encoder_lf0.to(device)
    encoder_spk.to(device)
    encoder_emo.to(device)

    classifier_spk.to(device)
    classifier_content.to(device)
    classifier_emo.to(device)
        
    # Load dataset
    root_path = Path(utils.to_absolute_path(cfg.data_root))
    print(root_path)
    dataset = CPCDataset(
        root=root_path,
        n_sample_frames=cfg.training.sample_frames,  # 128
        mode='test')

    # Load data
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,  # 30
        shuffle=True,
        num_workers=cfg.training.n_workers,
        pin_memory=True,
        drop_last=False)


    # Load checkpoint
    print("Load encoder checkpoint from: {}:".format(cfg.checkpoint_path))
    model_path = utils.to_absolute_path(cfg.checkpoint_path)
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    encoder.load_state_dict(checkpoint["encoder"])
    encoder_lf0.load_state_dict(checkpoint["encoder_lf0"])
    encoder_spk.load_state_dict(checkpoint["encoder_spk"])
    encoder_emo.load_state_dict(checkpoint["encoder_emo"])

    print("Load classifier checkpoint from: {}:".format(cfg.classifier_path))
    classifier_path = utils.to_absolute_path(cfg.classifier_path)
    classifiers = torch.load(classifier_path, map_location=lambda storage, loc: storage)
    classifier_spk.load_state_dict(classifiers["classifier_spk"])
    classifier_content.load_state_dict(classifiers["classifier_content"])
    classifier_emo.load_state_dict(classifiers["classifier_emo"])

    # Save training info
    if os.path.exists(f'{str(result_dir)}/results.txt'):
        wmode = 'a'
    else:
        wmode = 'w'
    results_txt = open(f'{str(result_dir)}/results.txt', wmode)
    results_txt.write('save training info...\n')
    results_txt.close()

    # Initialize accumulators before the loop
    total_correct_spk = 0
    total_correct_content = 0
    total_correct_emo = 0
    total_samples = 0

    for i, (labels, mels, lf0, speakers) in enumerate(dataloader, 1):
        # Ignore samples where speaker is 8 or 9
        mask = ~((speakers == 8) | (speakers == 9))
        if mask.sum() == 0:
            continue  # Skip this batch if all are class 8 or 9

        # Mask all batch tensors
        labels = labels[mask]
        mels = mels[mask]
        lf0 = lf0[mask]
        speakers = speakers[mask]

        lf0 = lf0.to(device)
        mels = mels.to(device)  # (bs, 80, 128)
        labels = labels.to(device)  # (bs, 128)

        z, c, _, vq_loss, perplexity = encoder(mels)
        spk_embs = encoder_spk(mels)
        lf0_embs = encoder_lf0(lf0)
        emo_embs = encoder_emo(mels)

        z_last = z[:, -1, :]

        speaker_logits = classifier_spk(spk_embs)
        content_logits = classifier_content(z_last)
        emotion_logits = classifier_emo(emo_embs)

        # Get predicted classes
        speaker_preds = speaker_logits.argmax(dim=1)
        content_preds = content_logits.argmax(dim=1)
        emotion_preds = emotion_logits.argmax(dim=1)

        # Accumulate correct counts and totals
        total_correct_spk += (speaker_preds == speakers).sum().item()
        total_correct_content += (content_preds == speakers).sum().item()
        total_correct_emo += (emotion_preds == speakers).sum().item()  

        total_samples += speakers.size(0)

    final_acc_spk = total_correct_spk / total_samples if total_samples > 0 else 0
    final_acc_content = total_correct_content / total_samples if total_samples > 0 else 0
    final_acc_emo = total_correct_emo / total_samples if total_samples > 0 else 0

    print(f"Final Speaker Accuracy: {final_acc_spk*100:.4f}%")
    print(f"Final Content Accuracy: {final_acc_content*100:.4f}%")
    print(f"Final Emotion Accuracy: {final_acc_emo*100:.4f}%")

    # Save training results to file
    results_txt = open(f'{str(result_dir)}/results.txt', 'a')
    results_txt.write(
        "Speaker accuracy:{}%, Content accuracy:{}%, Emotion accuracy:{:.3f}%"
        .format(final_acc_spk*100, final_acc_content*100, final_acc_emo*100))
    results_txt.close()



if __name__ == "__main__":
    train_model()
