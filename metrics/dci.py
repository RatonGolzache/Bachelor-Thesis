import hydra
from hydra import utils
import kaldiio
import numpy as np
np.int = int
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize


from dataset import CPCDataset_sameSeq as CPCDataset
from model_encoder import SpeakerEncoder, EmoEncoder

import scipy
from sklearn import ensemble

torch.cuda.empty_cache()
torch.manual_seed(137)
np.random.seed(137)

def disentanglement_per_code(importance_matrix):
    """Compute disentanglement score of each code."""
    # print("Disentanglement per code: importance_matrix shape:", importance_matrix.shape)
    # print("Disentanglement per code: importance_matrix sample values:\n", importance_matrix[:5, :])
    # Check for all zeros (could cause issues)
    # assert importance_matrix.shape[1] > 0, "importance_matrix has zero width"
    # assert not np.any(np.isnan(importance_matrix)), "NaN in importance_matrix"
    try:
        ent = scipy.stats.entropy(importance_matrix.T + 1e-11, base=importance_matrix.shape[1])
    except Exception as e:
        print("Exception in entropy calculation:", e)
        raise
    print("Disentanglement per code: entropy shape:", ent.shape)
    print("Disentanglement per code: entropy sample values:", ent[:5])
    return 1. - ent


def disentanglement(importance_matrix):
    """Compute the disentanglement score of the representation."""
    #print("Disentanglement: importance_matrix shape:", importance_matrix.shape)
    per_code = disentanglement_per_code(importance_matrix)
    if importance_matrix.sum() == 0.:
        print("Importance matrix sum is 0, replacing with ones")
        importance_matrix = np.ones_like(importance_matrix)
    code_importance = importance_matrix.sum(axis=1) / (importance_matrix.sum() + 1e-12)
    print("Disentanglement: code_importance shape:", code_importance.shape)
    print("Disentanglement: per_code shape:", per_code.shape)
    print("Disentanglement: code_importance sample values:", code_importance[:5])
    print("Disentanglement: per_code sample values:", per_code[:5])
    assert not np.any(np.isnan(per_code)), "NaN in per_code"
    assert not np.any(np.isnan(code_importance)), "NaN in code_importance"
    total = np.sum(per_code * code_importance)
    print("Disentanglement: total score:", total)
    return total

def compute_importance_single_gbt(x_train, y_train, x_test, y_test):
    """Compute feature importance for a single embedding with a single ground-truth factor."""
    num_codes = x_train.shape[0]
    importance_matrix = np.zeros(shape=[num_codes, 1], dtype=np.float64)
    model = ensemble.GradientBoostingClassifier()
    try:
        model.fit(x_train.T, y_train)
        importance_matrix[:, 0] = np.abs(model.feature_importances_)
        train_loss = np.mean(model.predict(x_train.T) == y_train)
        test_loss = np.mean(model.predict(x_test.T) == y_test)
    except Exception as e:
        print("Exception in GBT training:", e)
        raise
    return importance_matrix, train_loss, test_loss

def compute_importance_per_factor(x_train, y_train, x_test, y_test, num_factors):
    print("Computing importance per factor:")
    num_codes = x_train.shape[0]
    print("  Number of codes (features):", num_codes)
    print("  Number of factors:", num_factors)
    importance_matrix = np.zeros(shape=[num_codes, num_factors], dtype=np.float64)
    y_train_bin = label_binarize(y_train, classes=np.arange(num_factors))
    y_test_bin = label_binarize(y_test, classes=np.arange(num_factors))

    test_accuracies = []      # store test accuracies for averaging

    for i in range(num_factors):
        print(f"Training classifier for factor {i} / {num_factors}")
        model = ensemble.GradientBoostingClassifier()
        try:
            model.fit(x_train.T, y_train_bin[:, i])
            importance_matrix[:, i] = np.abs(model.feature_importances_)
            train_acc = np.mean(model.predict(x_train.T) == y_train_bin[:, i])
            test_acc = np.mean(model.predict(x_test.T) == y_test_bin[:, i])
            test_accuracies.append(test_acc)
            print(f"  Factor {i}: train acc {train_acc:.4f}, test acc {test_acc:.4f}")
            print(f"  Factor {i}: sample feature importances:", importance_matrix[:5, i])
        except Exception as e:
            print(f"Exception in training factor {i}:", e)
            raise

    avg_test_acc = np.mean(test_accuracies)
    print(f"Average test accuracy over {num_factors} factors: {avg_test_acc:.4f}")
    print("Completed feature importance computation for all factors.")
    print("Importance matrix shape:", importance_matrix.shape)
    print("Importance matrix sample values:\n", importance_matrix[:5, :])
    return importance_matrix, avg_test_acc


def extract_embeddings_and_labels(dataloader, encoder_spk, encoder_emo, device):
    spk_embeddings, spk_labels = [], []
    emo_embeddings, emo_labels = [], []
    with torch.no_grad():
        for i, (label, mel, lf0, speaker_idx) in enumerate(dataloader):
            mel = mel.to(device)
            # Speaker embedding and label
            spk_emb = encoder_spk(mel)
            spk_embeddings.append(spk_emb.cpu().numpy())
            spk_labels.append(speaker_idx.cpu().numpy())
            # Emotion embedding and label
            emo_emb = encoder_emo(mel)
            emo_embeddings.append(emo_emb.cpu().numpy())
            emo_labels.append(label.cpu().numpy())
    # concatenate over batch dimension
    spk_embeddings = np.concatenate(spk_embeddings, axis=0).T  # shape [dim_spk, num_samples]
    emo_embeddings = np.concatenate(emo_embeddings, axis=0).T  # shape [dim_emo, num_samples]
    spk_labels = np.concatenate(spk_labels).flatten()
    emo_labels = np.concatenate(emo_labels).flatten()
    print("Extracted embeddings and labels:")
    print(f"  Speaker embeddings shape: {spk_embeddings.shape}")
    print(f"  Emotion embeddings shape: {emo_embeddings.shape}")
    print(f"  Speaker labels shape: {spk_labels.shape}")
    print(f"  Emotion labels shape: {emo_labels.shape}")
    return spk_embeddings, spk_labels, emo_embeddings, emo_labels

def filter_test_by_seen_speakers(test_emb, test_spk_labels, train_spk_labels):
    seen_speakers = set(train_spk_labels)
    idx = np.isin(test_spk_labels, list(seen_speakers))
    return test_emb[:, idx], test_spk_labels[idx]

@hydra.main(config_path="./config/metrics.yaml")
def dci(cfg):
    root_path = Path(utils.to_absolute_path(cfg.data_root))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset_train = CPCDataset(root=root_path, n_sample_frames=cfg.training.sample_frames, mode='train')
    dataset_test = CPCDataset(root=root_path, n_sample_frames=cfg.training.sample_frames, mode='test')

    dataloader_train = DataLoader(dataset_train, batch_size=256, shuffle=True,
                                  num_workers=cfg.training.n_workers, pin_memory=True, drop_last=False)
    dataloader_test = DataLoader(dataset_test, batch_size=256, shuffle=False,
                                 num_workers=cfg.training.n_workers, pin_memory=True, drop_last=False)

    checkpoint_dir = Path(utils.to_absolute_path(cfg.checkpoint_dir))
    checkpoint_paths = sorted(checkpoint_dir.glob("*.pt"))  # adjust pattern if needed

    if not checkpoint_paths:
        print(f"No checkpoint files found in {checkpoint_dir}")
        return

    all_results = []

    for checkpoint_path in checkpoint_paths:
        checkpoint_stem = checkpoint_path.stem
        result_dir = Path(utils.to_absolute_path(f"dci_results_{checkpoint_stem}"))
        result_dir.mkdir(exist_ok=True, parents=True)

        print(f"\nProcessing checkpoint: {checkpoint_path}")
        encoder_spk = SpeakerEncoder().to(device)
        encoder_emo = EmoEncoder().to(device)

        checkpoint = torch.load(checkpoint_path, map_location=device)
        encoder_spk.load_state_dict(checkpoint["encoder_spk"])
        encoder_emo.load_state_dict(checkpoint["encoder_emo"])

        print("Extracting training embeddings and labels...")
        spk_emb_train, spk_lab_train, emo_emb_train, emo_lab_train = extract_embeddings_and_labels(dataloader_train, encoder_spk, encoder_emo, device)
        print("Extracting test embeddings and labels...")
        spk_emb_test, spk_lab_test, emo_emb_test, emo_lab_test = extract_embeddings_and_labels(dataloader_test, encoder_spk, encoder_emo, device)

        spk_emb_test, spk_lab_test = filter_test_by_seen_speakers(spk_emb_test, spk_lab_test, spk_lab_train)

        num_emo_factors = len(np.unique(emo_lab_test))
        num_spk_factors = len(np.unique(spk_lab_train))

        imp_spk, acc_spk_test = compute_importance_per_factor(spk_emb_train, spk_lab_train,
                                                              spk_emb_test, spk_lab_test, num_spk_factors)
        dis_spk = disentanglement(imp_spk)

        imp_emo, acc_emo_test = compute_importance_per_factor(emo_emb_train, emo_lab_train,
                                                              emo_emb_test, emo_lab_test, num_emo_factors)
        dis_emo = disentanglement(imp_emo)

        print(f"Checkpoint {checkpoint_stem}: Speaker disentanglement: {dis_spk:.4f}, test acc: {acc_spk_test:.4f}")
        print(f"Checkpoint {checkpoint_stem}: Emotion disentanglement: {dis_emo:.4f}, test acc: {acc_emo_test:.4f}")

        np.save(result_dir / "importance_spk.npy", imp_spk)
        np.save(result_dir / "importance_emo.npy", imp_emo)

        all_results.append({
            "model": checkpoint_stem,
            "speaker": {"disentanglement": dis_spk, "test_accuracy": acc_spk_test},
            "emotion": {"disentanglement": dis_emo, "test_accuracy": acc_emo_test},
        })

    combined_result_path = Path(utils.to_absolute_path("dci_results_combined_cremad"))
    combined_result_path.mkdir(exist_ok=True, parents=True)
    combined_txt_path = combined_result_path / "all_scores.txt"

    with open(combined_txt_path, "w") as f:
        for res in all_results:
            f.write(f"Model: {res['model']}\n")
            f.write(f"  Speaker disentanglement: {res['speaker']['disentanglement']:.6f}\n")
            f.write(f"  Speaker test accuracy: {res['speaker']['test_accuracy']:.6f}\n")
            f.write(f"  Emotion disentanglement: {res['emotion']['disentanglement']:.6f}\n")
            f.write(f"  Emotion test accuracy: {res['emotion']['test_accuracy']:.6f}\n")
            f.write("\n")

    print(f"\nAll results saved to {combined_txt_path}")

if __name__ == "__main__":
    dci()
