import os
import shutil
import random
from glob import glob

def organize_dataset(data_root, save_root, test_ratio=0.2):
    emotions = ["Angry", "Sad", "Surprise", "Happy", "Neutral"]
    os.makedirs(save_root, exist_ok=True)

    speakers = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]

    for speaker in speakers: 
        for split in ["train", "test"]:
            for emotion in emotions:
                os.makedirs(os.path.join(save_root, speaker, emotion, split), exist_ok=True)
    
    for speaker in speakers:
        for emotion in emotions:
            emotion_path = os.path.join(data_root, speaker, emotion)
            if not os.path.exists(emotion_path):
                continue  # Skip if the emotion folder doesn't exist
            
            files = glob(os.path.join(emotion_path, "*.wav"))
            if not files:
                continue
            
            random.shuffle(files)
            test_count = max(1, int(len(files) * test_ratio))  # Ensure at least one file goes to test
            
            test_files = files[:test_count]
            train_files = files[test_count:]
            
            for f in train_files:
                dest = os.path.join(save_root, speaker, emotion, "train", os.path.basename(f))
                shutil.copy2(f, dest)
            
            for f in test_files:
                dest = os.path.join(save_root, speaker, emotion, "test", os.path.basename(f))
                shutil.copy2(f, dest)
    
    print("Dataset reorganization complete.")

# Example usage
data_root = "./ESD/Emotion Speech Dataset"  # Path to the original dataset
save_root = "./ESD_Reorganized"  # Path where the reorganized data will be stored
organize_dataset(data_root, save_root)
