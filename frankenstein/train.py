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
from model_decoder import Decoder_ac
from model_encoder import VAEncoder

import apex.amp as amp
import os
import time

from utils_mlv import imshow_grid, mse_loss, reparameterize, group_wise_reparameterize, accumulate_group_evidence, group_wise_reparameterize_vectorized, accumulate_group_evidence_vectorized

torch.cuda.empty_cache()
torch.manual_seed(137)
np.random.seed(137)

# Function to save the model checkpoint
# This function saves the model's state_dict, optimizer state_dict, and other relevant information 
# to a specified directory. It creates the directory if it doesn't exist and saves the checkpoint with a 
# specific filename format. It also handles the case where mixed precision training is used by saving the 
# amp state_dict. 
def save_checkpoint(encoder, encoder_lf0, encoder_emo, cpc, encoder_spk, \
                    decoder, \
                    optimizer, \
                    scheduler, amp, epoch,
                    checkpoint_dir, cfg):
    if cfg.use_amp:
        amp_state_dict = amp.state_dict()
    else:
        amp_state_dict = None
    checkpoint_state = {
        "encoder": encoder.state_dict(),
        "encoder_lf0": encoder_lf0.state_dict(),
        "encoder_emo": encoder_emo.state_dict(),
        "cpc": cpc.state_dict(),
        "encoder_spk": encoder_spk.state_dict(),
        "decoder": decoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "amp": amp_state_dict,
        "epoch": epoch
    }
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    checkpoint_path = checkpoint_dir / "model.ckpt-{}.pt".format(epoch)
    torch.save(checkpoint_state, checkpoint_path)
    print("Saved checkpoint: {}".format(checkpoint_path.stem))


# This function performs a forward pass through the model, computes the losses, and updates the model parameters.
def forward(mels, lf0, encoder, encoder_emo, encoder_lf0, cpc, encoder_spk, decoder, cfg, optimizer, scheduler, labels, speakers):
    optimizer.zero_grad()
    #print('mels.shape:', mels.shape)
    # z = content embedding, c = context embedding (dependencies across time steps)
    z, c, _, vq_loss, perplexity = encoder(mels)
    cpc_loss, accuracy = cpc(z, c)
    lf0_embs = encoder_lf0(lf0)

    if mels.shape[-1] != 80:
        mel_batch = mels.permute(0, 2, 1).contiguous()
    else :
        mel_batch = mels

    spk_mu, spk_logvar = encoder_spk(mel_batch)
    emo_mu, emo_logvar = encoder_emo(mel_batch)

    spk_grouped_mu, spk_grouped_logvar = accumulate_group_evidence_vectorized(
                spk_mu, spk_logvar, speakers, False
    )

    # Up batch size
    # kl-divergence error for speaker latent space
    spk_kl_divergence_loss =  cfg.kl_divergence_coef * (
            - 0.5 * torch.sum(1 + spk_grouped_logvar - spk_grouped_mu.pow(2) - spk_grouped_logvar.exp())
    )
    spk_kl_divergence_loss /= (mels.shape[0] * 80 * 128)


    emo_grouped_mu, emo_grouped_logvar = accumulate_group_evidence_vectorized(
                emo_mu, emo_logvar, labels, True
    )
    # kl-divergence error for emotion latent space
    emo_kl_divergence_loss = cfg.kl_divergence_coef * (
            - 0.5 * torch.sum(1 + emo_grouped_logvar - emo_grouped_mu.pow(2) - emo_grouped_logvar.exp())
    )
    emo_kl_divergence_loss /= (mels.shape[0] * 80 * 128)

    spk_embs = group_wise_reparameterize_vectorized(
        training=True, mu=spk_grouped_mu, logvar=spk_grouped_logvar, labels_batch=speakers, cuda=True
    )

    # print('spk_embs.shape:', spk_embs.shape)

    emo_embs = group_wise_reparameterize_vectorized(
        training=True, mu=emo_grouped_mu, logvar=emo_grouped_logvar, labels_batch=labels, cuda=True
    )

    # print('emo_embs.shape:', emo_embs.shape)

    #print('spk_embs.shape:', spk_embs.shape)
    #print('emo_embs.shape:', emo_embs.shape)

    recon_loss, pred_mels = decoder(z, lf0_embs, spk_embs, emo_embs, mels.transpose(1, 2))

    # Compute the losses
    # recon_loss = reconstruction loss of the decoder
    # cpc_loss = contrastive predictive coding loss when predicting the future frames of a sequence given the past frames
    # vq_loss = vector quantization loss
    # spk_kl_divergence_loss = KL divergence loss for the speaker latent space
    # emo_kl_divergence_loss = KL divergence loss for the emotion latent space
    loss = recon_loss + cpc_loss + vq_loss + spk_kl_divergence_loss + emo_kl_divergence_loss

    # Perform backpropagation and optimization
    if cfg.use_amp:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()

    #  ---- Examine gradients before optimizer step ----
    #print("==== Gradient Norms Before Optimizer Step ====")
    #for name, param in encoder_spk.named_parameters():
    #    if param.requires_grad and param.grad is not None:
    #        grad_norm = param.grad.data.norm(2).item()
    #        print(f"[Speaker Encoder] {name} grad norm: {grad_norm:.6f}")
    #    elif param.requires_grad:
    #        print(f"[Speaker Encoder] {name} has NO grad!")

    #for name, param in encoder_emo.named_parameters():
    #    if param.requires_grad and param.grad is not None:
    #        grad_norm = param.grad.data.norm(2).item()
    #        print(f"[Emotion Encoder] {name} grad norm: {grad_norm:.6f}")
    #    elif param.requires_grad:
    #        print(f"[Emotion Encoder] {name} has NO grad!")

    optimizer.step()
    return optimizer, recon_loss, cpc_loss, vq_loss, accuracy, perplexity, spk_kl_divergence_loss, emo_kl_divergence_loss


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
    encoder_spk = VAEncoder(cfg.classifier_num_speakers) # speaker encoder
    encoder_emo = VAEncoder(cfg.classifier_num_classes) # style encoder

    decoder = Decoder_ac(dim_neck=cfg.model.encoder.z_dim, dim_lf0=dim_lf0, use_l1_loss=True)

    encoder.to(device)
    cpc.to(device)
    encoder_lf0.to(device)
    encoder_spk.to(device)

    encoder_emo.to(device)

    decoder.to(device)

    optimizer = optim.Adam(
        chain(encoder.parameters(), encoder_lf0.parameters(), cpc.parameters(), encoder_spk.parameters(),
              encoder_emo.parameters(), decoder.parameters()),
        lr=cfg.training.scheduler.initial_lr)

    # TODO: use_amp is set default to True to speed up training; no-amp -> more stable training? => need to be verified
    if cfg.use_amp:
        [encoder, encoder_lf0, cpc, encoder_spk, decoder], optimizer = amp.initialize(
            [encoder, encoder_lf0, cpc, encoder_spk, decoder], optimizer, opt_level='O1')
        
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
    if cfg.resume:
        print("Resume checkpoint from: {}:".format(cfg.resume))
        resume_path = utils.to_absolute_path(cfg.resume)
        checkpoint = torch.load(resume_path, map_location=lambda storage, loc: storage)
        encoder.load_state_dict(checkpoint["encoder"])
        encoder_lf0.load_state_dict(checkpoint["encoder_lf0"])
        cpc.load_state_dict(checkpoint["cpc"])
        encoder_spk.load_state_dict(checkpoint["encoder_spk"])
        decoder.load_state_dict(checkpoint["decoder"])
        optimizer.load_state_dict(checkpoint["optimizer"])
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
        average_cpc_loss = average_vq_loss = average_perplexity = average_recon_loss = 0
        average_emo_kl_divergence_loss = average_spk_kl_divergence_loss = 0
        average_accuracies = np.zeros(cfg.training.n_prediction_steps)


        # Train
        for i, (labels, mels, lf0, speakers) in enumerate(dataloader, 1):
            lf0 = lf0.to(device)
            mels = mels.to(device)  # (bs, 80, 128)
            labels = labels.to(device)  # (bs, 1)
            speakers = speakers.to(device)  # (bs, 1)

            optimizer, recon_loss, cpc_loss, vq_loss, accuracy, perplexity, spk_kl_divergence_loss, emo_kl_divergence_loss = forward(
                mels, lf0, encoder, encoder_emo, encoder_lf0, cpc, encoder_spk, decoder, 
                cfg, optimizer, scheduler, labels, speakers)
            
            # Update the average losses and accuracies
            average_recon_loss += (recon_loss.item() - average_recon_loss) / i
            average_cpc_loss += (cpc_loss.item() - average_cpc_loss) / i
            average_vq_loss += (vq_loss.item() - average_vq_loss) / i
            average_perplexity += (perplexity.item() - average_perplexity) / i
            average_accuracies += (np.array(accuracy) - average_accuracies) / i
            average_spk_kl_divergence_loss += (spk_kl_divergence_loss.item() - average_spk_kl_divergence_loss) / i
            average_emo_kl_divergence_loss += (emo_kl_divergence_loss.item() - average_emo_kl_divergence_loss) / i


            # Print training progress
            ctime = time.time()
            if (global_step % 25 == 0) and (global_step != 0):
                print(
                    "epoch:{}, global step:{}, recon loss:{:.3f}, cpc loss:{:.3f}, vq loss:{:.3f}, perpexlity:{:.3f}, emotion encoder kl divergence loss:{:.8f}, speaker encoder kl divergence loss:{:.8f}, used time:{:.3f}s"
                        .format(epoch, global_step, average_recon_loss, average_cpc_loss, average_vq_loss,
                                average_perplexity, average_emo_kl_divergence_loss, average_spk_kl_divergence_loss, ctime - stime))
                print(100 * average_accuracies)
            stime = time.time()
            global_step += 1
            # scheduler.step()

        # Save training results to file
        results_txt = open(f'{str(checkpoint_dir)}/results.txt', 'a')
        results_txt.write(
            "epoch:{}, global step:{}, recon loss:{:.3f}, cpc loss:{:.3f}, vq loss:{:.3f}, perpexlity:{:.3f}, emotion encoder kl divergence loss:{:.8f}, speaker encoder kl divergence loss:{:.8f}"
            .format(epoch, global_step, average_recon_loss, average_cpc_loss, average_vq_loss, average_perplexity, average_emo_kl_divergence_loss, average_spk_kl_divergence_loss) + '\n')
        results_txt.write(' '.join([str(cpc_acc) for cpc_acc in average_accuracies]) + '\n')
        results_txt.close()
        scheduler.step()

       
        # Save checkpoint
        if epoch % cfg.training.checkpoint_interval == 0 and epoch != start_epoch:
            save_checkpoint(encoder, encoder_lf0, encoder_emo, cpc, encoder_spk, \
                            decoder, \
                            optimizer, \
                            scheduler, amp, epoch,
                            checkpoint_dir, cfg)


if __name__ == "__main__":
    train_model()
