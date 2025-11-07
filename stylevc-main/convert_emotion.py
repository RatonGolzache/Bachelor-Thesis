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

    # Load checkpoints
    checkpoint = torch.load(cfg.checkpoint, map_location=lambda storage, loc: storage)
    encoder.load_state_dict(checkpoint["encoder"])
    encoder_spk.load_state_dict(checkpoint["encoder_spk"])
    encoder_emo.load_state_dict(checkpoint["encoder_emo"])
    encoder_lf0.load_state_dict(checkpoint["encoder_lf0"])
    decoder.load_state_dict(checkpoint["decoder"])

    encoder.eval()
    encoder_emo.eval()
    encoder_spk.eval()
    decoder.eval()

    # Directory structure
    base_dir = Path(r"D:/Bsc-Thesis/stylevc-main/data_esd/test")
    speaker = "0011"

    # Gather one utterance per emotion
    emotions = [d.name for d in base_dir.iterdir() if d.is_dir()]
    utterances = {}
    for emotion in emotions:
        mel_dir = base_dir / emotion / "mels" / speaker
        f0_dir = base_dir / emotion / "lf0" / speaker
        mel_files = sorted(mel_dir.glob("*.npy"))
        f0_files = sorted(f0_dir.glob("*.npy"))
        if mel_files and f0_files:
            code = mel_files[0].stem  # e.g., '0011_000001'
            utterances[emotion] = {
                "mel": mel_files[0],
                "f0": f0_dir / f"{code}.npy"
            }

    # Convert every emotion to every other emotion
    out_dir = Path(cfg.out_path)
    out_dir.mkdir(exist_ok=True, parents=True)

    for target_emotion, target_files in utterances.items():
        for ref_emotion, ref_files in utterances.items():
            if target_emotion == ref_emotion:
                continue  # skip self-to-self

            # Load features
            target_mel = np.load(target_files["mel"])
            target_f0 = np.load(target_files["f0"])
            ref_mel = np.load(ref_files["mel"])
            ref_f0 = np.load(ref_files["f0"])

            # Prepare tensors
            target_mel_tensor = torch.FloatTensor(target_mel.T).unsqueeze(0).to(device)
            target_f0_tensor = torch.FloatTensor(target_f0).unsqueeze(0).to(device)
            ref_mel_tensor = torch.FloatTensor(ref_mel.T).unsqueeze(0).to(device)
            ref_f0_tensor = torch.FloatTensor(ref_f0).unsqueeze(0).to(device)

            with torch.no_grad():
                z, _, _, _ = encoder.encode(target_mel_tensor)
                lf0_embs = encoder_lf0(target_f0_tensor)
                spk_embs = encoder_spk(target_mel_tensor)
                emo_embs = encoder_emo(ref_mel_tensor)
                output = decoder(z, lf0_embs, spk_embs, emo_embs)
                logmel = output.squeeze(0).cpu().numpy()

            # Save mel-spectrogram
            out_filename = f"{speaker}_{target_emotion}_to_{ref_emotion}"
            feat_writer = kaldiio.WriteHelper(
                f"ark,scp:{out_dir}/{out_filename}.ark,{out_dir}/{out_filename}.scp"
            )
            feat_writer[out_filename] = logmel
            feat_writer.close()

            # Synthesize waveform with vocoder
            cmd = [
                'parallel-wavegan-decode',
                '--checkpoint', cfg.vocoder_path,
                '--feats-scp', f'{out_dir}/{out_filename}.scp',
                '--outdir', str(out_dir)
            ]
            subprocess.call(cmd)

if __name__ == "__main__":
    convert()
