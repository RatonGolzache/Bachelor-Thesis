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
from scheduler import WarmupScheduler
from model_encoder import Encoder, CPCLoss_sameSeq, Encoder_lf0
from model_encoder import EmoEncoder
from model_decoder import Decoder_ac
from model_encoder import SpeakerEncoder as Encoder_spk
from model_encoder import ClassifierSpk, ClassifierContent, ClassifierEmo

import apex.amp as amp
import os
import time

torch.cuda.empty_cache()
torch.manual_seed(137)
np.random.seed(137)

# Function to save the model checkpoint
# This function saves the model's state_dict, optimizer state_dict, and other relevant information 
# to a specified directory. It creates the directory if it doesn't exist and saves the checkpoint with a 
# specific filename format. It also handles the case where mixed precision training is used by saving the 
# amp state_dict. 
def save_checkpoint(classifier_emo, classifier_spk, classifier_content, scheduler, amp, epoch,
                    checkpoint_dir, cfg):
    if cfg.use_amp:
        amp_state_dict = amp.state_dict()
    else:
        amp_state_dict = None
    checkpoint_state = {
        "classifier_spk": classifier_spk.state_dict(),
        "classifier_content": classifier_content.state_dict(),
        "classifier_emo": classifier_emo.state_dict(),
        "scheduler": scheduler.state_dict(),
        "amp": amp_state_dict,
        "epoch": epoch
    }
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    checkpoint_path = checkpoint_dir / "model.ckpt-{}.pt".format(epoch)
    torch.save(checkpoint_state, checkpoint_path)
    print("Saved checkpoint: {}".format(checkpoint_path.stem))

@hydra.main(config_path="./config/train.yaml")
def train_model(cfg):
    
    #cfg.checkpoint_dir = f'{cfg.checkpoint_dir}'
    if cfg.encoder_lf0_type == 'no_emb':  # default
        dim_lf0 = 1
    else:
        dim_lf0 = 64

    checkpoint_dir = Path(utils.to_absolute_path("checkpoint_spk_classifiers_label_forward_no_mi"))
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # define model
    encoder = Encoder(**cfg.model.encoder) # config -> model -> default.yaml
    encoder_lf0 = Encoder_lf0(cfg.encoder_lf0_type) # no_emb
    encoder_spk = Encoder_spk() # speaker encoder
    encoder_emo = EmoEncoder() # style encoder

    classifier_spk = ClassifierSpk(256, 128, cfg.classifier_num_speakers) # classifier from speaker embedding
    classifier_content = ClassifierContent(64, 32, cfg.classifier_num_speakers) # classifier from content embedding
    classifier_emo = ClassifierEmo(256, 128, cfg.classifier_num_speakers) # classifier from style embedding

    encoder.to(device)
    encoder_lf0.to(device)
    encoder_spk.to(device)
    encoder_emo.to(device)

    classifier_spk.to(device)
    classifier_content.to(device)
    classifier_emo.to(device)

    # TODO: Here Need to be Changed
    optimizer = optim.Adam(
        chain(classifier_spk.parameters(), classifier_content.parameters(), classifier_emo.parameters()),
        lr=cfg.training.scheduler.initial_lr)

    # TODO: use_amp is set default to True to speed up training; no-amp -> more stable training? => need to be verified
    if cfg.use_amp:
        [classifier_spk, classifier_content, classifier_emo], optimizer = amp.initialize(
            [classifier_spk, classifier_content, classifier_emo], optimizer, opt_level='O1')
        
    # Load dataset
    root_path = Path(utils.to_absolute_path(cfg.data_root))
    print(root_path)
    dataset = CPCDataset(
        root=root_path,
        n_sample_frames=cfg.training.sample_frames,  # 128
        mode='train')


    # Warmup scheduler
    warmup_epochs = 2000 // (len(dataset) // cfg.training.batch_size)
    print('warmup_epochs:', warmup_epochs)
    scheduler = WarmupScheduler(
        optimizer,
        warmup_epochs=warmup_epochs,
        initial_lr=cfg.training.scheduler.initial_lr,
        max_lr=cfg.training.scheduler.max_lr,
        milestones=cfg.training.scheduler.milestones,
        gamma=cfg.training.scheduler.gamma)

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
    resume_path = utils.to_absolute_path(cfg.checkpoint_path)
    checkpoint = torch.load(resume_path, map_location=lambda storage, loc: storage)
    encoder.load_state_dict(checkpoint["encoder"])
    encoder_lf0.load_state_dict(checkpoint["encoder_lf0"])
    encoder_spk.load_state_dict(checkpoint["encoder_spk"])
    encoder_emo.load_state_dict(checkpoint["encoder_emo"])
    start_epoch = 1


    # Save training info
    if os.path.exists(f'{str(checkpoint_dir)}/results.txt'):
        wmode = 'a'
    else:
        wmode = 'w'
    results_txt = open(f'{str(checkpoint_dir)}/results.txt', wmode)
    results_txt.write('save training info...\n')
    results_txt.close()

    global_step = 0
    stime = time.time()

    # Training loop
    for epoch in range(start_epoch, cfg.training.n_epochs + 1):
        average_cross_entropy_loss_spk = average_cross_entropy_loss_content = average_cross_entropy_loss_emo = 0.0

        # Train
        for i, (labels, mels, lf0, speakers) in enumerate(dataloader, 1):
            lf0 = lf0.to(device)
            mels = mels.to(device)  # (bs, 80, 128)
            speakers = speakers.to(device)  # (bs, 128)

            optimizer.zero_grad()

            z, c, _, vq_loss, perplexity = encoder(mels)
            spk_embs = encoder_spk(mels)
            lf0_embs = encoder_lf0(lf0)
            emo_embs = encoder_emo(mels)

            z_last = z[:, -1, :]

            speaker_logits = classifier_spk(spk_embs)
            content_logits = classifier_content(z_last)
            emotion_logits = classifier_emo(emo_embs)

            cross_entropy_loss_spk = nn.CrossEntropyLoss()(speaker_logits, speakers)
            cross_entropy_loss_content = nn.CrossEntropyLoss()(content_logits, speakers)
            cross_entropy_loss_emo = nn.CrossEntropyLoss()(emotion_logits, speakers)

            loss = cross_entropy_loss_spk + cross_entropy_loss_content + cross_entropy_loss_emo

            # Perform backpropagation and optimization
            if cfg.use_amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            optimizer.step()

            # Update the average losses and accuracies
            average_cross_entropy_loss_spk += (cross_entropy_loss_spk.item() - average_cross_entropy_loss_spk) / i
            average_cross_entropy_loss_content += (cross_entropy_loss_content.item() - average_cross_entropy_loss_content) / i
            average_cross_entropy_loss_emo += (cross_entropy_loss_emo.item() - average_cross_entropy_loss_emo) / i

            # Print training progress
            ctime = time.time()
            if (global_step % 25 == 0) and (global_step != 0):
                print(
                    "epoch:{}, global step:{}, speaker cross entropy:{:.3f}, content cross entropy:{:.3f}, emotion cross entropy:{:.3f}, used time:{:.3f}s"
                        .format(epoch, global_step, average_cross_entropy_loss_spk, average_cross_entropy_loss_content, average_cross_entropy_loss_emo, ctime - stime))
            stime = time.time()
            global_step += 1
            # scheduler.step()

        # Save training results to file
        results_txt = open(f'{str(checkpoint_dir)}/results.txt', 'a')
        results_txt.write(
            "epoch:{}, global step:{}, speaker cross entropy:{:.3f}, content cross entropy:{:.3f}, emotion cross entropy:{:.3f}\n"
            .format(epoch, global_step, average_cross_entropy_loss_spk, average_cross_entropy_loss_content, average_cross_entropy_loss_emo))
        results_txt.close()
        scheduler.step()

        # Save checkpoint
        if epoch % cfg.training.checkpoint_interval == 0 and epoch != start_epoch:
            save_checkpoint(classifier_emo, classifier_spk, classifier_content, scheduler, amp, epoch, checkpoint_dir, cfg)


if __name__ == "__main__":
    train_model()
