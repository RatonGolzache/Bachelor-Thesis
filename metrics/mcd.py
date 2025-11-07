import hydra
from hydra import utils
import kaldiio
import numpy as np
np.int = int
from pathlib import Path
import torch
from torch.utils.data import DataLoader


from dataset import CPCDataset_sameSeq as CPCDataset
from model_encoder import Encoder, Encoder_lf0, EmoEncoder, SpeakerEncoder
from model_decoder import Decoder_ac
from mel_cepstral_distance import compare_mel_spectrograms

torch.cuda.empty_cache()
torch.manual_seed(137)
np.random.seed(137)


@hydra.main(config_path="./config/metrics.yaml")
def mcd(cfg):
    root_path = Path(utils.to_absolute_path(cfg.data_root_mcd))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset_test = CPCDataset(root=root_path, n_sample_frames=cfg.training.sample_frames, mode='test')
    dataloader_test = DataLoader(dataset_test, batch_size=256, shuffle=False,
                                 num_workers=cfg.training.n_workers, pin_memory=True, drop_last=False)

    checkpoint_dir = Path(utils.to_absolute_path(cfg.checkpoint_dir_mcd))
    checkpoint_paths = sorted(checkpoint_dir.glob("*.pt"))  # adjust pattern if needed

    if not checkpoint_paths:
        print(f"No checkpoint files found in {checkpoint_dir}")
        return

    all_results = []

    for checkpoint_path in checkpoint_paths:
        checkpoint_stem = checkpoint_path.stem

        print(f"\nProcessing checkpoint: {checkpoint_path}")
        # Load models
        encoder = Encoder(**cfg.model.encoder).to(device)
        encoder_lf0 = Encoder_lf0().to(device)
        encoder_spk = SpeakerEncoder().to(device)
        encoder_emo = EmoEncoder().to(device)
        decoder = Decoder_ac(dim_neck=64).to(device)


        checkpoint = torch.load(checkpoint_path, map_location=device)
        encoder.load_state_dict(checkpoint["encoder"])
        encoder_lf0.load_state_dict(checkpoint["encoder_lf0"])
        encoder_spk.load_state_dict(checkpoint["encoder_spk"])
        encoder_emo.load_state_dict(checkpoint["encoder_emo"])
        decoder.load_state_dict(checkpoint["decoder"])

        mcd_vals = []
        with torch.no_grad():
            for i, (_, mels, lf0, _) in enumerate(dataloader_test):
                mels.to(device)
                lf0.to(device)

                z, _, _, _ = encoder.encode(mels)
                lf0_embs = encoder_lf0(lf0)
                spk_embs= encoder_spk(mels)
                emo_embs = encoder_emo(mels)
                output = decoder(z, lf0_embs, spk_embs, emo_embs)
                logmel_pred = output.squeeze(0).cpu().numpy()
                logmel_true = mels.squeeze(0).cpu().numpy()
                logmel_true = logmel_true.transpose(0, 2, 1) 
                #print(logmel_pred.shape, logmel_true.shape)

                # Batch-wise MCD computation
                for mel_pred, mel_true in zip(logmel_pred, logmel_true):
                    #print(mel_pred.shape, mel_true.shape)
                    mcd, _ = compare_mel_spectrograms(mel_true, mel_pred)
                    mcd_vals.append(mcd)

        avg_mcd = float(np.mean(mcd_vals)) if mcd_vals else None
        print(f"Average MCD for {checkpoint_stem}: {avg_mcd}")

        all_results.append({
            "model": checkpoint_stem,
            "avg_mcd": avg_mcd,
        })

    combined_result_path = Path(utils.to_absolute_path("mcd_results_combined_cremad"))
    combined_result_path.mkdir(exist_ok=True, parents=True)
    combined_txt_path = combined_result_path / "all_scores.txt"

    with open(combined_txt_path, "w") as f:
        for res in all_results:
            f.write(f"Model: {res['model']}\n")
            f.write(f"Average MCD: {res['avg_mcd']}\n")
            f.write("\n")

    print(f"\nAll results saved to {combined_txt_path}")

if __name__ == "__main__":
    mcd()
