import hydra
import torch
import numpy as np
from pathlib import Path
from model_encoder import Encoder, Encoder_lf0, EmoEncoder, SpeakerEncoder as Encoder_spk
from model_decoder import Decoder_ac
import kaldiio
import subprocess

@hydra.main(config_path="./config/convert.yaml")
def convert(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models
    encoder = Encoder(**cfg.model.encoder).to(device)
    encoder_lf0 = Encoder_lf0().to(device)
    encoder_spk = Encoder_spk().to(device)
    encoder_emo = EmoEncoder().to(device)
    decoder = Decoder_ac(dim_neck=64).to(device)

    encoder2 = Encoder(**cfg.model.encoder).to(device)
    encoder_lf02 = Encoder_lf0().to(device)
    encoder_spk2 = Encoder_spk().to(device)
    encoder_emo2 = EmoEncoder().to(device)
    decoder2 = Decoder_ac(dim_neck=64).to(device)

    # Load checkpoints
    # checkpoint = torch.load("D:\Bsc-Thesis\checkpoint-stylevc-cremad\model.ckpt-750.pt", map_location=lambda storage, loc: storage)
    checkpoint = torch.load("D:\Bsc-Thesis\stylevc-main\checkpoint_label_forward_no_mi\model_with_classifier.ckpt-500.pt", map_location=lambda storage, loc: storage)
    encoder.load_state_dict(checkpoint["encoder"])
    encoder_spk.load_state_dict(checkpoint["encoder_spk"])
    encoder_emo.load_state_dict(checkpoint["encoder_emo"])
    encoder_lf0.load_state_dict(checkpoint["encoder_lf0"])
    decoder.load_state_dict(checkpoint["decoder"])

    checkpoint2 = torch.load("D:\Bsc-Thesis\stylevc-main\checkpoint_data_esd_neutral\model.ckpt-500.pt", map_location=lambda storage, loc: storage)
    encoder2.load_state_dict(checkpoint2["encoder"])
    encoder_spk2.load_state_dict(checkpoint2["encoder_spk"])
    encoder_emo2.load_state_dict(checkpoint2["encoder_emo"])
    encoder_lf02.load_state_dict(checkpoint2["encoder_lf0"])
    decoder2.load_state_dict(checkpoint2["decoder"])

    encoder.eval()
    encoder_emo.eval()
    encoder_spk.eval()
    decoder.eval()

    # Load target and reference features from config paths
    target_mel = np.load(cfg.target_mel_path)  # shape: (T, 80)
    target_f0 = np.load(cfg.target_f0_path)    # shape: (T,) or (T, 1)
    #ref_mel = np.load(cfg.ref_mel_path)        # shape: (T, 80)
    #ref_f0 = np.load(cfg.ref_f0_path)          # shape: (T,) or (T, 1)
    ref_mel_2 = np.load(cfg.ref_mel_path_2)        # shape: (T, 80)
    ref_f0_2 = np.load(cfg.ref_f0_path_2)          # shape: (T,) or (T, 1)
    #ref_mel_2 = np.load(cfg.ref_mel_path_neu)        # shape: (T, 80)
    #ref_f0_2 = np.load(cfg.ref_f0_path_neu)          # shape: (T,) or (T, 1)
    #ref_mel = np.load(cfg.ref_mel_path_3)        # shape: (T, 80)
    #ref_f0 = np.load(cfg.ref_f0_path_3)          # shape: (T,) or (T, 1)
    #ref_mel = np.load(cfg.ref_mel_path_4)        # shape: (T, 80)
    #ref_f0 = np.load(cfg.ref_f0_path_4)          # shape: (T,) or (T, 1)
    #ref_mel = np.load(cfg.ref_mel_path_5)        # shape: (T, 80)
    #ref_f0 = np.load(cfg.ref_f0_path_5)          # shape: (T,) or (T, 1)


    # target_mel = np.load(cfg.cremad_target_mel_path)  # shape: (T, 80)
    # target_f0 = np.load(cfg.cremad_target_f0_path)    # shape: (T,) or (T, 1)
    #ref_mel = np.load(cfg.cremad_ref_mel_path)        # shape: (T, 80)
    #ref_f0 = np.load(cfg.cremad_ref_f0_path)          # shape: (T,) or (T, 1)
    # ref_mel = np.load(cfg.cremad_ref_mel_path_2)        # shape: (T, 80)
    # ref_f0 = np.load(cfg.cremad_ref_f0_path_2)          # shape: (T,) or (T, 1)

    ref_mel = np.load(cfg.ref_mel_path_ang_3)        # shape: (T, 80)
    ref_f0 = np.load(cfg.ref_f0_path_ang_3)          # shape: (T,) or (T, 1)  

    # Prepare tensors
    target_mel_tensor = torch.FloatTensor(target_mel.T).unsqueeze(0).to(device)  # (1, 80, T)
    target_f0_tensor = torch.FloatTensor(target_f0).unsqueeze(0).to(device)      # (1, T) or (1, T, 1)
    ref_mel_tensor = torch.FloatTensor(ref_mel.T).unsqueeze(0).to(device)        # (1, 80, T)
    ref_f0_tensor = torch.FloatTensor(ref_f0).unsqueeze(0).to(device)            # (1, T) or (1, T, 1)
    ref_mel_tensor_2 = torch.FloatTensor(ref_mel_2.T).unsqueeze(0).to(device)        # (1, 80, T)
    ref_f0_tensor_2 = torch.FloatTensor(ref_f0_2).unsqueeze(0).to(device)            # (1, T) or (1, T, 1)


    with torch.no_grad():
        # Content encoding from target
        z, _, _, _ = encoder.encode(target_mel_tensor)
        z_ref, _, _, _ = encoder.encode(ref_mel_tensor)
        # Pitch encoding from target
        lf0_embs = encoder_lf0(target_f0_tensor)
        lf0_embs_ref = encoder_lf0(ref_f0_tensor)
        # Speaker embedding from target
        spk_embs_target = encoder_spk(target_mel_tensor)
        spk_embs_ref = encoder_spk(ref_mel_tensor)
        # Emotion embedding from reference
        emo_embs_ref = encoder_emo(ref_mel_tensor)
        emo_embs_target = encoder_emo(target_mel_tensor)
        emo_embs_ref2 = encoder_emo2(ref_mel_tensor_2)
        # Decode with swapped emotion
        output = decoder(z, lf0_embs, spk_embs_target, emo_embs_ref)
        
        logmel = output.squeeze(0).cpu().numpy()

    # Save mel-spectrogram
    out_dir = Path(cfg.out_path)
    out_dir.mkdir(exist_ok=True, parents=True)
    out_filename = Path(cfg.target_mel_path).stem
    feat_writer = kaldiio.WriteHelper(f"ark,scp:{out_dir}/feats.1.ark,{out_dir}/feats.1.scp")
    feat_writer[out_filename] = logmel
    feat_writer.close()

    # Synthesize waveform with vocoder
    cmd = [
        'parallel-wavegan-decode',
        '--checkpoint', cfg.vocoder_path,
        '--feats-scp', f'{str(out_dir)}/feats.1.scp',
        '--outdir', str(out_dir)
    ]
    subprocess.call(cmd)

if __name__ == "__main__":
    convert()
