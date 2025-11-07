import os
import shutil
import random
from glob import glob

def organize_dataset(data_root, save_root):

    emotions = ["ANG", "SAD", "DIS", "HAP", "NEU", "FEA"]
    os.makedirs(save_root, exist_ok=True)

    # Get all wav files
    wav_files = [f for f in os.listdir(data_root) if f.endswith(".wav")]

    # Organize by speaker and emotion
    speaker_emotion_dict = {}
    for wav in wav_files:
        parts = wav.split("_")
        speaker, emotion = parts[0], parts[2]
        if speaker not in speaker_emotion_dict:
            speaker_emotion_dict[speaker] = {e: [] for e in emotions}
        speaker_emotion_dict[speaker][emotion].append(wav)

    # Move files into the new structure
    for speaker, emotion_dict in speaker_emotion_dict.items():
        for emotion, files in emotion_dict.items():
            if not files:
                continue
            
            # Create directories
            train_dir = os.path.join(save_root, speaker, emotion, "train")
            test_dir = os.path.join(save_root, speaker, emotion, "test")
            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(test_dir, exist_ok=True)
            
            # Select test set
            test_files = random.sample(files, max(1, len(files) // 10))  # 10% test data
            train_files = [f for f in files if f not in test_files]
            
            # Move files
            for file in train_files:
                shutil.copy(os.path.join(data_root, file), os.path.join(train_dir, file))
            for file in test_files:
                shutil.copy(os.path.join(data_root, file), os.path.join(test_dir, file))

    print("Dataset reorganized successfully!")


# Paths
source_folder = "./CREMA-D/AudioWAV"  # Folder containing all wav files
save_root = "./CREMA-D_Reorganized"  # Target dataset structure

organize_dataset(source_folder, save_root)
