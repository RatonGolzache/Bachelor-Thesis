import hydra
from hydra import utils
from itertools import chain
from pathlib import Path
import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import CPCDataset_sameSeq as CPCDataset
from scheduler import WarmupScheduler
from model_encoder import Encoder, CPCLoss_sameSeq, Encoder_lf0
from model_encoder import EmoEncoder
from model_decoder import Decoder_ac
from model_encoder import SpeakerEncoder as Encoder_spk
from mi_estimators import CLUBSample_group, CLUBSample_reshape

import torch.nn as nn
from model_encoder import SpeakerClassifier


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
def save_checkpoint(encoder, encoder_lf0, encoder_emo, cpc, encoder_spk, \
                    cs_mi_net, ps_mi_net, cp_mi_net, ep_mi_net, ec_mi_net, se_mi_net, decoder, \
                    optimizer, optimizer_cs_mi_net, optimizer_ps_mi_net, optimizer_cp_mi_net, \
                    optimizer_ep_mi_net, optimizer_ec_mi_net, optimizer_se_mi_net, scheduler, amp, epoch,
                    checkpoint_dir, cfg, spk_classifier):
    if cfg.use_amp:
        amp_state_dict = amp.state_dict()
    else:
        amp_state_dict = None
    checkpoint_state = {
        "encoder": encoder.state_dict(),
        "encoder_lf0": encoder_lf0.state_dict(),
        "encoder_emo": encoder_emo.state_dict(),
        "spk_classifier": spk_classifier.state_dict(), 
        "cpc": cpc.state_dict(),
        "encoder_spk": encoder_spk.state_dict(),
        "ps_mi_net": ps_mi_net.state_dict(),
        "cp_mi_net": cp_mi_net.state_dict(),
        "cs_mi_net": cs_mi_net.state_dict(),
        "ep_mi_net": ep_mi_net.state_dict(),
        "ec_mi_net": ec_mi_net.state_dict(),
        "se_mi_net": se_mi_net.state_dict(),
        "decoder": decoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "optimizer_cs_mi_net": optimizer_cs_mi_net.state_dict(),
        "optimizer_ps_mi_net": optimizer_ps_mi_net.state_dict(),
        "optimizer_cp_mi_net": optimizer_cp_mi_net.state_dict(),
        "optimizer_ec_mi_net": optimizer_ec_mi_net.state_dict(),
        "optimizer_ep_mi_net": optimizer_ep_mi_net.state_dict(),
        "optimizer_se_mi_net": optimizer_se_mi_net.state_dict(),
        "scheduler": scheduler.state_dict(),
        "amp": amp_state_dict,
        "epoch": epoch
    }
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    checkpoint_path = checkpoint_dir / "model_with_classifier.ckpt-{}.pt".format(epoch)
    torch.save(checkpoint_state, checkpoint_path)
    print("Saved checkpoint: {}".format(checkpoint_path.stem))

# Train MI estimators individually using log-likelihood
def mi_first_forward(mels, lf0, encoder, encoder_lf0, encoder_spk, encoder_emo, cs_mi_net, optimizer_cs_mi_net,
                     ps_mi_net, optimizer_ps_mi_net, cp_mi_net, optimizer_cp_mi_net, ec_mi_net, optimizer_ec_mi_net,
                     se_mi_net, optimizer_se_mi_net, ep_mi_net, optimizer_ep_mi_net, cfg):
    optimizer_cs_mi_net.zero_grad()
    optimizer_ps_mi_net.zero_grad()
    optimizer_cp_mi_net.zero_grad()
    optimizer_ec_mi_net.zero_grad()
    optimizer_se_mi_net.zero_grad()
    optimizer_ep_mi_net.zero_grad()

    # z = content embedding
    z, _, _, _, _ = encoder(mels)
    z = z.detach()
    lf0_embs = encoder_lf0(lf0).detach()
    spk_embs = encoder_spk(mels).detach()
    emo_embs = encoder_emo(mels).detach()
    # print(f"lf0_embs shape: {lf0_embs.shape}, spk_embs shape: {spk_embs.shape}, emo_embs shape: {emo_embs.shape}")

    if cfg.use_CSMI:
        lld_cs_loss = -cs_mi_net.loglikeli(spk_embs, z)
        if cfg.use_amp:
            with amp.scale_loss(lld_cs_loss, optimizer_cs_mi_net) as sl:
                sl.backward()
        else:
            lld_cs_loss.backward()
        optimizer_cs_mi_net.step()
    else:
        lld_cs_loss = torch.tensor(0.)

    if cfg.use_CPMI:
        lld_cp_loss = -cp_mi_net.loglikeli(
            lf0_embs.unsqueeze(1).reshape(lf0_embs.shape[0], -1, 2, lf0_embs.shape[-1]).mean(2), z)
        if cfg.use_amp:
            with amp.scale_loss(lld_cp_loss, optimizer_cp_mi_net) as slll:
                slll.backward()
        else:
            lld_cp_loss.backward()
        torch.nn.utils.clip_grad_norm_(cp_mi_net.parameters(), 1)
        optimizer_cp_mi_net.step()
    else:
        lld_cp_loss = torch.tensor(0.)

    if cfg.use_PSMI:
        lld_ps_loss = -ps_mi_net.loglikeli(spk_embs, lf0_embs)
        if cfg.use_amp:
            with amp.scale_loss(lld_ps_loss, optimizer_ps_mi_net) as sll:
                sll.backward()
        else:
            lld_ps_loss.backward()
        optimizer_ps_mi_net.step()
    else:
        lld_ps_loss = torch.tensor(0.)

    if cfg.use_ECMI:
        lld_ec_loss = -ec_mi_net.loglikeli(emo_embs, z)
        if cfg.use_amp:
            with amp.scale_loss(lld_ec_loss, optimizer_ec_mi_net) as sl:
                sl.backward()
        else:
            lld_ec_loss.backward()
        optimizer_ec_mi_net.step()
    else:
        lld_ec_loss = torch.tensor(0.)

    if cfg.use_SEMI:
        lld_se_loss = -se_mi_net.loglikeli(spk_embs, emo_embs.unsqueeze(2))
        if cfg.use_amp:
            with amp.scale_loss(lld_se_loss, optimizer_se_mi_net) as slll:
                slll.backward()
        else:
            lld_se_loss.backward()
        optimizer_se_mi_net.step()
    else:
        lld_se_loss = torch.tensor(0.)

    if cfg.use_EPMI:
        lld_ep_loss = -ep_mi_net.loglikeli(emo_embs, lf0_embs)
        if cfg.use_amp:
            with amp.scale_loss(lld_ep_loss, optimizer_ep_mi_net) as sll:
                sll.backward()
        else:
            lld_ep_loss.backward()
        optimizer_ep_mi_net.step()
    else:
        lld_ep_loss = torch.tensor(0.)

    return optimizer_cs_mi_net, lld_cs_loss, optimizer_ps_mi_net, lld_ps_loss, optimizer_cp_mi_net, lld_cp_loss, \
           optimizer_ec_mi_net, lld_ec_loss, optimizer_se_mi_net, lld_se_loss, optimizer_ep_mi_net, lld_ep_loss

# Train the model using the MI estimators and the other losses
# This function performs a forward pass through the model, computes the losses, and updates the model parameters.
def mi_second_forward(mels, lf0, encoder, encoder_emo, encoder_lf0, cpc, encoder_spk, cs_mi_net, ps_mi_net, cp_mi_net,ec_mi_net, se_mi_net, ep_mi_net, decoder, cfg, optimizer, scheduler, spk_classifier, speakers):
    optimizer.zero_grad()
    # z = content embedding, c = context embedding (dependencies across time steps)
    z, c, _, vq_loss, perplexity = encoder(mels)
    cpc_loss, accuracy = cpc(z, c)
    spk_embs = encoder_spk(mels)
    lf0_embs = encoder_lf0(lf0)
    emo_embs = encoder_emo(mels)
    recon_loss, pred_mels = decoder(z, lf0_embs, spk_embs, emo_embs, mels.transpose(1, 2))

    speaker_logits = spk_classifier(spk_embs)
    cross_entropy_loss = nn.CrossEntropyLoss()(speaker_logits, speakers)

    # Compute the losses
    # recon_loss = reconstruction loss of the decoder
    # cpc_loss = contrastive predictive coding loss when predicting the future frames of a sequence given the past frames
    # vq_loss = vector quantization loss
    loss = recon_loss + cpc_loss + vq_loss + cross_entropy_loss

    if cfg.use_CSMI:
        mi_cs_loss = cfg.mi_weight * cs_mi_net.mi_est(spk_embs, z)
    else:
        mi_cs_loss = torch.tensor(0.).to(loss.device)

    if cfg.use_CPMI:
        mi_cp_loss = cfg.mi_weight * cp_mi_net.mi_est(
            lf0_embs.unsqueeze(1).reshape(lf0_embs.shape[0], -1, 2, lf0_embs.shape[-1]).mean(2), z)
    else:
        mi_cp_loss = torch.tensor(0.).to(loss.device)

    if cfg.use_PSMI:
        mi_ps_loss = cfg.mi_weight * ps_mi_net.mi_est(spk_embs, lf0_embs)
    else:
        mi_ps_loss = torch.tensor(0.).to(loss.device)

    if cfg.use_ECMI:
        mi_ec_loss = cfg.mi_weight * ec_mi_net.mi_est(emo_embs, z)
    else:
        mi_ec_loss = torch.tensor(0.).to(loss.device)

    if cfg.use_SEMI:
        mi_se_loss = cfg.mi_weight * se_mi_net.mi_est(spk_embs, emo_embs.unsqueeze(2))
    else:
        mi_se_loss = torch.tensor(0.).to(loss.device)

    if cfg.use_EPMI:
        mi_ep_loss = cfg.mi_weight * ep_mi_net.mi_est(emo_embs, lf0_embs)
    else:
        mi_ep_loss = torch.tensor(0.).to(loss.device)

    # Add the MI losses to the total loss
    # The style loss is weighted by 2, as the emotional style representation carries more correlation information
    loss = loss + mi_cs_loss + mi_ps_loss + mi_cp_loss + 2*mi_ec_loss + 2*mi_se_loss + 2*mi_ep_loss

    # Perform backpropagation and optimization
    if cfg.use_amp:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()

    optimizer.step()
    return optimizer, recon_loss, vq_loss, cpc_loss, accuracy, perplexity, mi_cs_loss, mi_ps_loss, mi_cp_loss, mi_ec_loss, mi_se_loss, mi_ep_loss, cross_entropy_loss


def calculate_eval_loss(mels, lf0, \
                        encoder, encoder_lf0, cpc, \
                        encoder_spk, encoder_emo, cs_mi_net, ps_mi_net, \
                        cp_mi_net, ec_mi_net, se_mi_net, ep_mi_net, decoder, cfg, spk_classifier, speakers):
    with torch.no_grad():
        z, c, z_beforeVQ, vq_loss, perplexity = encoder(mels)
        c = c
        lf0_embs = encoder_lf0(lf0)
        spk_embs = encoder_spk(mels)
        emo_embs = encoder_emo(mels)
        # print(f"lf0_embs shape: {lf0_embs.shape}, spk_embs shape: {spk_embs.shape}, emo_embs shape: {emo_embs.shape}")

        cpc_loss, accuracy = cpc(z, c)
        recon_loss, pred_mels = decoder(z, lf0_embs, spk_embs, emo_embs, mels.transpose(1, 2))

        speaker_logits = spk_classifier(spk_embs)
        cross_entropy_loss = nn.CrossEntropyLoss()(speaker_logits, speakers)

        if cfg.use_CSMI:
            lld_cs_loss = -cs_mi_net.loglikeli(spk_embs, z)
            mi_cs_loss = cfg.mi_weight * cs_mi_net.mi_est(spk_embs, z)
        else:
            lld_cs_loss = torch.tensor(0.)
            mi_cs_loss = torch.tensor(0.)

        if cfg.use_CPMI:
            mi_cp_loss = cfg.mi_weight * cp_mi_net.mi_est(
                lf0_embs.unsqueeze(1).reshape(lf0_embs.shape[0], -1, 2, lf0_embs.shape[-1]).mean(2), z)
            lld_cp_loss = -cp_mi_net.loglikeli(
                lf0_embs.unsqueeze(1).reshape(lf0_embs.shape[0], -1, 2, lf0_embs.shape[-1]).mean(2), z)
        else:
            mi_cp_loss = torch.tensor(0.)
            lld_cp_loss = torch.tensor(0.)

        if cfg.use_PSMI:
            mi_ps_loss = cfg.mi_weight * ps_mi_net.mi_est(spk_embs, lf0_embs)
            lld_ps_loss = -ps_mi_net.loglikeli(spk_embs, lf0_embs)
        else:
            mi_ps_loss = torch.tensor(0.)
            lld_ps_loss = torch.tensor(0.)

        if cfg.use_ECMI:
            lld_ec_loss = -ec_mi_net.loglikeli(emo_embs, z)
            mi_ec_loss = cfg.mi_weight * ec_mi_net.mi_est(emo_embs, z)
        else:
            lld_ec_loss = torch.tensor(0.)
            mi_ec_loss = torch.tensor(0.)

        if cfg.use_SEMI:
            mi_se_loss = cfg.mi_weight * se_mi_net.mi_est(spk_embs, emo_embs.unsqueeze(2))
            lld_se_loss = -se_mi_net.loglikeli(spk_embs, emo_embs.unsqueeze(2))
        else:
            mi_se_loss = torch.tensor(0.)
            lld_se_loss = torch.tensor(0.)

        if cfg.use_EPMI:
            mi_ep_loss = cfg.mi_weight * ep_mi_net.mi_est(emo_embs, lf0_embs)
            lld_ep_loss = -ep_mi_net.loglikeli(emo_embs, lf0_embs)
        else:
            mi_ep_loss = torch.tensor(0.)
            lld_ep_loss = torch.tensor(0.)

        return recon_loss, vq_loss, cpc_loss, accuracy, perplexity, mi_cs_loss, lld_cs_loss, mi_ps_loss, lld_ps_loss, \
               mi_cp_loss, lld_cp_loss, mi_ec_loss, lld_ec_loss, mi_se_loss, lld_se_loss, mi_ep_loss, lld_ep_loss, cross_entropy_loss


def to_eval(all_models):
    for m in all_models:
        m.eval()


def to_train(all_models):
    for m in all_models:
        m.train()


def eval_model(epoch, checkpoint_dir, device, valid_dataloader, encoder, encoder_lf0, encoder_emo, cpc, encoder_spk,
               cs_mi_net, ps_mi_net, cp_mi_net, ec_mi_net, se_mi_net, ep_mi_net, decoder, cfg, spk_classifier):
    stime = time.time()
    average_cpc_loss = average_vq_loss = average_perplexity = average_recon_loss = average_cross_entropy_loss = 0
    average_accuracies = np.zeros(cfg.training.n_prediction_steps)
    average_lld_cs_loss = average_mi_cs_loss = average_lld_ps_loss = average_mi_ps_loss = average_lld_cp_loss = average_mi_cp_loss = 0
    average_lld_ec_loss = average_mi_ec_loss = average_lld_se_loss = average_mi_se_loss = average_lld_ep_loss = average_mi_ep_loss = 0
    all_models = [encoder, encoder_lf0, cpc, encoder_spk, encoder_emo, cs_mi_net, ps_mi_net, cp_mi_net, ec_mi_net,
                  se_mi_net, ep_mi_net, decoder, spk_classifier]
    to_eval(all_models)
    for i, (labels, mels, lf0, speakers) in enumerate(valid_dataloader, 1):
        lf0 = lf0.to(device)
        mels = mels.to(device)  # (bs, 80, 128)
        speakers = speakers.to(device)  # (bs, 1)
        recon_loss, vq_loss, cpc_loss, accuracy, perplexity, mi_cs_loss, lld_cs_loss, mi_ps_loss, lld_ps_loss, \
        mi_cp_loss, lld_cp_loss, mi_ec_loss, lld_ec_loss, mi_se_loss, lld_se_loss, mi_ep_loss, lld_ep_loss, cross_entropy_loss = \
            calculate_eval_loss(mels, lf0, \
                                encoder, encoder_lf0, cpc, \
                                encoder_spk, encoder_emo, cs_mi_net, ps_mi_net, \
                                cp_mi_net, ec_mi_net, se_mi_net, ep_mi_net, decoder, cfg, spk_classifier, speakers)

        average_cross_entropy_loss += (cross_entropy_loss.item() - average_cross_entropy_loss) / i
        average_recon_loss += (recon_loss.item() - average_recon_loss) / i
        average_cpc_loss += (cpc_loss.item() - average_cpc_loss) / i
        average_vq_loss += (vq_loss.item() - average_vq_loss) / i
        average_perplexity += (perplexity.item() - average_perplexity) / i
        average_accuracies += (np.array(accuracy) - average_accuracies) / i
        average_lld_cs_loss += (lld_cs_loss.item() - average_lld_cs_loss) / i
        average_mi_cs_loss += (mi_cs_loss.item() - average_mi_cs_loss) / i
        average_lld_ps_loss += (lld_ps_loss.item() - average_lld_ps_loss) / i
        average_mi_ps_loss += (mi_ps_loss.item() - average_mi_ps_loss) / i
        average_lld_cp_loss += (lld_cp_loss.item() - average_lld_cp_loss) / i
        average_mi_cp_loss += (mi_cp_loss.item() - average_mi_cp_loss) / i
        average_lld_ec_loss += (lld_ec_loss.item() - average_lld_ec_loss) / i
        average_mi_ec_loss += (mi_ec_loss.item() - average_mi_ec_loss) / i
        average_lld_se_loss += (lld_se_loss.item() - average_lld_se_loss) / i
        average_mi_se_loss += (mi_se_loss.item() - average_mi_se_loss) / i
        average_lld_ep_loss += (lld_ep_loss.item() - average_lld_ep_loss) / i
        average_mi_ep_loss += (mi_ep_loss.item() - average_mi_ep_loss) / i

    ctime = time.time()
    print(
        "Eval | epoch:{}, cross entropy loss:{:.3f}, recon loss:{:.3f}, cpc loss:{:.3f}, vq loss:{:.3f}, perpexlity:{:.3f}, lld cs loss:{:.3f}, mi cs loss:{:.3E}, lld ps loss:{:.3f}, mi ps loss:{:.3f}, lld cp loss:{:.3f}, mi cp loss:{:.3f}, used time:{:.3f}s"
            .format(epoch, average_cross_entropy_loss, average_recon_loss, average_cpc_loss, average_vq_loss, average_perplexity,
                    average_lld_cs_loss,
                    average_mi_cs_loss, average_lld_ps_loss, average_mi_ps_loss, average_lld_cp_loss,
                    average_mi_cp_loss,
                    average_lld_ec_loss, average_mi_ec_loss, average_lld_se_loss, average_mi_se_loss,
                    average_lld_ep_loss, average_mi_ep_loss,
                    ctime - stime))
    print(100 * average_accuracies)
    results_txt = open(f'{str(checkpoint_dir)}/results.txt', 'a')
    results_txt.write(
        "Eval | epoch:{}, cross entropy loss:{:.3f}, recon loss:{:.3f}, cpc loss:{:.3f}, vq loss:{:.3f}, perpexlity:{:.3f}, lld cs loss:{:.3f}, mi cs loss:{:.3E}, lld ps loss:{:.3f}, mi ps loss:{:.3f}, lld cp loss:{:.3f}, mi cp loss:{:.3f}"
        .format(epoch, average_cross_entropy_loss, average_recon_loss, average_cpc_loss, average_vq_loss, average_perplexity, average_lld_cs_loss,
                average_mi_cs_loss, average_lld_ps_loss, average_mi_ps_loss, average_lld_cp_loss, average_mi_cp_loss,
                average_lld_ec_loss, average_mi_ec_loss, average_lld_se_loss, average_mi_se_loss, average_lld_ep_loss,
                average_mi_ep_loss) + '\n')
    results_txt.write(' '.join([str(cpc_acc) for cpc_acc in average_accuracies]) + '\n')
    results_txt.close()

    to_train(all_models)


@hydra.main(config_path="./config/train.yaml")
def train_model(cfg):
    
    #cfg.checkpoint_dir = f'{cfg.checkpoint_dir}'
    if cfg.encoder_lf0_type == 'no_emb':  # default
        dim_lf0 = 1
    else:
        dim_lf0 = 64

    checkpoint_dir = Path(utils.to_absolute_path(cfg.checkpoint_dir))
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # define model
    encoder = Encoder(**cfg.model.encoder) # config -> model -> default.yaml
    encoder_lf0 = Encoder_lf0(cfg.encoder_lf0_type) # no_emb
    cpc = CPCLoss_sameSeq(**cfg.model.cpc) # config -> model -> default.yaml
    encoder_spk = Encoder_spk() # speaker encoder

    encoder_emo = EmoEncoder() # style encoder

    spk_classifier = SpeakerClassifier(cfg.classifier_input_dim, cfg.classifier_num_speakers) # speaker classifier

    # mi_estimators

    # content / speaker
    cs_mi_net = CLUBSample_group(256, cfg.model.encoder.z_dim, 512)
    # pitch / speaker
    ps_mi_net = CLUBSample_group(256, dim_lf0, 512)
    # content / pitch
    cp_mi_net = CLUBSample_reshape(dim_lf0, cfg.model.encoder.z_dim, 512)

    # speaker / emotion
    se_mi_net = CLUBSample_group(256, dim_lf0, 512)
    # emotion / content
    ec_mi_net = CLUBSample_group(256, cfg.model.encoder.z_dim, 512)
    # emotion / pitch
    ep_mi_net = CLUBSample_group(256, dim_lf0, 512)

    decoder = Decoder_ac(dim_neck=cfg.model.encoder.z_dim, dim_lf0=dim_lf0, use_l1_loss=True)

    encoder.to(device)
    cpc.to(device)
    encoder_lf0.to(device)
    encoder_spk.to(device)

    encoder_emo.to(device)

    spk_classifier.to(device)

    cs_mi_net.to(device)
    ps_mi_net.to(device)
    cp_mi_net.to(device)
    se_mi_net.to(device)
    ec_mi_net.to(device)
    ep_mi_net.to(device)
    decoder.to(device)
    # TODO: Here Need to be Changed
    optimizer = optim.Adam(
        chain(encoder.parameters(), encoder_lf0.parameters(), cpc.parameters(), encoder_spk.parameters(),
              encoder_emo.parameters(), decoder.parameters(), spk_classifier.parameters()),
        lr=cfg.training.scheduler.initial_lr)
    optimizer_cs_mi_net = optim.Adam(cs_mi_net.parameters(), lr=cfg.mi_lr)
    optimizer_ps_mi_net = optim.Adam(ps_mi_net.parameters(), lr=cfg.mi_lr)
    optimizer_cp_mi_net = optim.Adam(cp_mi_net.parameters(), lr=cfg.mi_lr)
    optimizer_se_mi_net = optim.Adam(se_mi_net.parameters(), lr=cfg.mi_lr)
    optimizer_ec_mi_net = optim.Adam(ec_mi_net.parameters(), lr=cfg.mi_lr)
    optimizer_ep_mi_net = optim.Adam(ep_mi_net.parameters(), lr=cfg.mi_lr)
    # TODO: use_amp is set default to True to speed up training; no-amp -> more stable training? => need to be verified
    if cfg.use_amp:
        [encoder, encoder_lf0, cpc, encoder_spk, decoder, spk_classifier], optimizer = amp.initialize(
            [encoder, encoder_lf0, cpc, encoder_spk, decoder, spk_classifier], optimizer, opt_level='O1')
        [cs_mi_net], optimizer_cs_mi_net = amp.initialize([cs_mi_net], optimizer_cs_mi_net, opt_level='O1')
        [ps_mi_net], optimizer_ps_mi_net = amp.initialize([ps_mi_net], optimizer_ps_mi_net, opt_level='O1')
        [cp_mi_net], optimizer_cp_mi_net = amp.initialize([cp_mi_net], optimizer_cp_mi_net, opt_level='O1')
        [se_mi_net], optimizer_se_mi_net = amp.initialize([se_mi_net], optimizer_se_mi_net, opt_level='O1')
        [ec_mi_net], optimizer_ec_mi_net = amp.initialize([ec_mi_net], optimizer_ec_mi_net, opt_level='O1')
        [ep_mi_net], optimizer_ep_mi_net = amp.initialize([ep_mi_net], optimizer_ep_mi_net, opt_level='O1')

    # Load dataset
    root_path = Path(utils.to_absolute_path(cfg.data_root))
    print(root_path)
    dataset = CPCDataset(
        root=root_path,
        n_sample_frames=cfg.training.sample_frames,  # 128
        mode='train')
    valid_dataset = CPCDataset(
        root=root_path,
        n_sample_frames=cfg.training.sample_frames,  # 128
        mode='valid')

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
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=cfg.training.batch_size,  # 30
        shuffle=False,
        num_workers=cfg.training.n_workers,
        pin_memory=True,
        drop_last=False)

    # Load checkpoint
    if cfg.resume:
        print("Resume checkpoint from: {}:".format(cfg.resume))
        resume_path = utils.to_absolute_path(cfg.resume)
        checkpoint = torch.load(resume_path, map_location=lambda storage, loc: storage)
        encoder.load_state_dict(checkpoint["encoder"])
        encoder_lf0.load_state_dict(checkpoint["encoder_lf0"])
        cpc.load_state_dict(checkpoint["cpc"])
        encoder_spk.load_state_dict(checkpoint["encoder_spk"])
        spk_classifier.load_state_dict(checkpoint["spk_classifier"])
        cs_mi_net.load_state_dict(checkpoint["cs_mi_net"])
        ps_mi_net.load_state_dict(checkpoint["ps_mi_net"])
        if cfg.use_CPMI:
            cp_mi_net.load_state_dict(checkpoint["cp_mi_net"])
        if cfg.use_ECMI:
            ec_mi_net.load_state_dict(checkpoint["ec_mi_net"])
        if cfg.use_SEMI:
            se_mi_net.load_state_dict(checkpoint["se_mi_net"])
        if cfg.use_EPMI:
            ep_mi_net.load_state_dict(checkpoint["ep_mi_net"])
        decoder.load_state_dict(checkpoint["decoder"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        optimizer_cs_mi_net.load_state_dict(checkpoint["optimizer_cs_mi_net"])
        optimizer_ps_mi_net.load_state_dict(checkpoint["optimizer_ps_mi_net"])
        optimizer_cp_mi_net.load_state_dict(checkpoint["optimizer_cp_mi_net"])
        optimizer_ec_mi_net.load_state_dict(checkpoint["optimizer_ec_mi_net"])
        optimizer_se_mi_net.load_state_dict(checkpoint["optimizer_se_mi_net"])
        optimizer_ep_mi_net.load_state_dict(checkpoint["optimizer_ep_mi_net"])
        if cfg.use_amp:
            amp.load_state_dict(checkpoint["amp"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"]
    else:
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
        average_cpc_loss = average_vq_loss = average_perplexity = average_recon_loss = average_cross_entropy_loss = 0
        average_accuracies = np.zeros(cfg.training.n_prediction_steps)
        average_lld_cs_loss = average_mi_cs_loss = average_lld_ps_loss = average_mi_ps_loss = average_lld_cp_loss = average_mi_cp_loss = 0
        average_lld_ec_loss = average_mi_ec_loss = average_lld_se_loss = average_mi_se_loss = average_lld_ep_loss = average_mi_ep_loss = 0

        # Train
        for i, (labels, mels, lf0, speakers) in enumerate(dataloader, 1):
            lf0 = lf0.to(device)
            mels = mels.to(device)  # (bs, 80, 128)
            speakers = speakers.to(device)  # (bs, 1)
            # If using speaker MI loss, perform the first forward pass to estimate log-likelihoods and train the estimators
            if cfg.use_CSMI or cfg.use_CPMI or cfg.use_PSMI:
                for j in range(cfg.mi_iters):
                    optimizer_cs_mi_net, lld_cs_loss, optimizer_ps_mi_net, lld_ps_loss, optimizer_cp_mi_net, lld_cp_loss, \
                    optimizer_ec_mi_net, lld_ec_loss, optimizer_se_mi_net, lld_se_loss, optimizer_ep_mi_net, lld_ep_loss = mi_first_forward(
                        mels, lf0, encoder, encoder_lf0, encoder_spk, encoder_emo, cs_mi_net, optimizer_cs_mi_net,
                        ps_mi_net, optimizer_ps_mi_net, cp_mi_net, optimizer_cp_mi_net, ec_mi_net, optimizer_ec_mi_net,
                        se_mi_net, optimizer_se_mi_net, ep_mi_net, optimizer_ep_mi_net, cfg)
            else:
                lld_cs_loss = torch.tensor(0.)
                lld_ps_loss = torch.tensor(0.)
                lld_cp_loss = torch.tensor(0.)

            # Perform the second forward pass to compute the main loss and update the model parameters
            optimizer, recon_loss, vq_loss, cpc_loss, accuracy, perplexity, mi_cs_loss, \
            mi_ps_loss, mi_cp_loss, mi_ec_loss, mi_se_loss, mi_ep_loss, cross_entropy_loss = mi_second_forward(
                mels, lf0, encoder, encoder_emo, encoder_lf0,cpc, encoder_spk, cs_mi_net, ps_mi_net, cp_mi_net,\
                ec_mi_net, se_mi_net, ep_mi_net, decoder, cfg, optimizer, scheduler, spk_classifier, speakers)

            # Update the average losses and accuracies
            average_cross_entropy_loss += (cross_entropy_loss.item() - average_cross_entropy_loss) / i
            average_recon_loss += (recon_loss.item() - average_recon_loss) / i
            average_cpc_loss += (cpc_loss.item() - average_cpc_loss) / i
            average_vq_loss += (vq_loss.item() - average_vq_loss) / i
            average_perplexity += (perplexity.item() - average_perplexity) / i
            average_accuracies += (np.array(accuracy) - average_accuracies) / i
            average_lld_cs_loss += (lld_cs_loss.item() - average_lld_cs_loss) / i
            average_mi_cs_loss += (mi_cs_loss.item() - average_mi_cs_loss) / i
            average_lld_ps_loss += (lld_ps_loss.item() - average_lld_ps_loss) / i
            average_mi_ps_loss += (mi_ps_loss.item() - average_mi_ps_loss) / i
            average_lld_cp_loss += (lld_cp_loss.item() - average_lld_cp_loss) / i
            average_mi_cp_loss += (mi_cp_loss.item() - average_mi_cp_loss) / i
            average_lld_ec_loss += (lld_ec_loss.item() - average_lld_ec_loss) / i
            average_mi_ec_loss += (mi_ec_loss.item() - average_mi_ec_loss) / i
            average_lld_se_loss += (lld_se_loss.item() - average_lld_se_loss) / i
            average_mi_se_loss += (mi_se_loss.item() - average_mi_se_loss) / i
            average_lld_ep_loss += (lld_ep_loss.item() - average_lld_ep_loss) / i
            average_mi_ep_loss += (mi_ep_loss.item() - average_mi_ep_loss) / i

            # Print training progress
            ctime = time.time()
            if (global_step % 25 == 0) and (global_step != 0):
                print(
                    "epoch:{}, global step:{}, cross entropy loss:{:.3f}, recon loss:{:.3f}, cpc loss:{:.3f}, vq loss:{:.3f}, perpexlity:{:.3f}, lld cs loss:{:.3f}, mi cs loss:{:.3E}, lld ps loss:{:.3f}, mi ps loss:{:.3f}, lld cp loss:{:.3f}, mi cp loss:{:.3f}, used time:{:.3f}s"
                        .format(epoch, global_step, average_cross_entropy_loss, average_recon_loss, average_cpc_loss, average_vq_loss,
                                average_perplexity,
                                average_lld_cs_loss, average_mi_cs_loss, average_lld_ps_loss, average_mi_ps_loss,
                                average_lld_cp_loss, average_mi_cp_loss, average_lld_ec_loss, average_mi_ec_loss, \
                                average_lld_se_loss, average_mi_se_loss, average_lld_ep_loss, average_mi_ep_loss,
                                ctime - stime))
                print(100 * average_accuracies)
            stime = time.time()
            global_step += 1
            # scheduler.step()

        # Save training results to file
        results_txt = open(f'{str(checkpoint_dir)}/results.txt', 'a')
        results_txt.write(
            "epoch:{}, global step:{}, cross entropy loss:{:.3f}, recon loss:{:.3f}, cpc loss:{:.3f}, vq loss:{:.3f}, perpexlity:{:.3f}, lld cs loss:{:.3f}, mi cs loss:{:.3E}, lld ps loss:{:.3f}, mi ps loss:{:.3f}, lld cp loss:{:.3f}, mi cp loss:{:.3f}"
            .format(epoch, global_step, average_cross_entropy_loss, average_recon_loss, average_cpc_loss, average_vq_loss, average_perplexity,
                    average_lld_cs_loss, average_mi_cs_loss, average_lld_ps_loss, average_mi_ps_loss,
                    average_lld_cp_loss, average_mi_cp_loss, average_lld_ec_loss, average_mi_ec_loss, \
                    average_lld_se_loss, average_mi_se_loss, average_lld_ep_loss, average_mi_ep_loss) + '\n')
        results_txt.write(' '.join([str(cpc_acc) for cpc_acc in average_accuracies]) + '\n')
        results_txt.close()
        scheduler.step()

        # Evaluate checkpoints
        if epoch % cfg.training.log_interval == 0 and epoch != start_epoch:
            eval_model(epoch, checkpoint_dir, device, valid_dataloader, encoder, encoder_lf0, encoder_emo, cpc,
                       encoder_spk, cs_mi_net, ps_mi_net, cp_mi_net, ec_mi_net, se_mi_net, ep_mi_net, decoder, cfg, spk_classifier)

            ctime = time.time()
            print(
                "epoch:{}, global step:{}, cross entropy loss:{:.3f}, recon loss:{:.3f}, cpc loss:{:.3f}, vq loss:{:.3f}, perpexlity:{:.3f}, lld cs loss:{:.3f}, mi cs loss:{:.3E}, lld ps loss:{:.3f}, mi ps loss:{:.3f}, lld cp loss:{:.3f}, mi cp loss:{:.3f}, used time:{:.3f}s"
                    .format(epoch, global_step, average_cross_entropy_loss, average_recon_loss, average_cpc_loss, average_vq_loss,
                            average_perplexity,
                            average_lld_cs_loss, average_mi_cs_loss, average_lld_ps_loss, average_mi_ps_loss,
                            average_lld_cp_loss, average_mi_cp_loss, average_lld_ec_loss, average_mi_ec_loss, \
                            average_lld_se_loss, average_mi_se_loss, average_lld_ep_loss, average_mi_ep_loss,
                            ctime - stime))
            print(100 * average_accuracies)
            stime = time.time()

        # Save checkpoint
        if epoch % cfg.training.checkpoint_interval == 0 and epoch != start_epoch:
            save_checkpoint(encoder, encoder_lf0, encoder_emo, cpc, encoder_spk, \
                            cs_mi_net, ps_mi_net, cp_mi_net, ep_mi_net, ec_mi_net, se_mi_net, decoder, \
                            optimizer, optimizer_cs_mi_net, optimizer_ps_mi_net, optimizer_cp_mi_net, \
                            optimizer_ep_mi_net, optimizer_ec_mi_net, optimizer_se_mi_net, scheduler, amp, epoch,
                            checkpoint_dir, cfg, spk_classifier)


if __name__ == "__main__":
    train_model()
