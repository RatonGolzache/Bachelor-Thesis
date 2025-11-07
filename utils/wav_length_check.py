import os
import librosa
from collections import defaultdict

def calculate_duration_percentage_with_emotion_distribution(directories, emotions):
    # Duration bins in seconds
    bins = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]
    
    # Initialize counters for each emotion in each bin
    emotion_counts = {emotion: [0] * len(bins) for emotion in emotions}
    bin_counts = [0] * len(bins)  # Initialize counts for each bin
    total_files = 0

    # Iterate over each directory
    for directory in directories:
        for file in os.listdir(directory):  # List files in the directory (no subdirectories)
            if file.endswith('.wav'):  # Check if it's a .wav file
                total_files += 1
                wav_path = os.path.join(directory, file)
                
                try:
                    # Load the WAV file and calculate its duration
                    y, sr = librosa.load(wav_path, sr=None)  # Load with original sample rate
                    duration = librosa.get_duration(y=y, sr=sr)
                    
                    # Check which bin the duration falls into
                    for i, (lower, upper) in enumerate(bins):
                        if lower <= duration < upper:
                            bin_counts[i] += 1
                            # Check for each emotion and increment the corresponding count if it's in the path
                            for emotion in emotions:
                                if emotion in wav_path:  # Emotion is in the file path
                                    emotion_counts[emotion][i] += 1
                            break
                except Exception as e:
                    print(f"Error processing {wav_path}: {e}")
    
    # Calculate percentages for each bin and each emotion
    emotion_percentages = {emotion: [] for emotion in emotions}
    for emotion in emotions:
        for i, count in enumerate(emotion_counts[emotion]):
            # If the bin count is > 0, calculate percentage, else set it to 0
            percentage = (count / bin_counts[i]) * 100 if bin_counts[i] > 0 else 0
            emotion_percentages[emotion].append(percentage)

    # Calculate total percentage for each bin
    bin_percentages = [(count / total_files) * 100 if total_files > 0 else 0 for count in bin_counts]
    
    # Print the results
    print(f"Total files: {total_files}")
    for i, (lower, upper) in enumerate(bins):
        print(f"Duration between {lower}s and {upper}s:")
        print(f"  Total percentage: {bin_percentages[i]:.2f}%")
        for emotion in emotions:
            print(f"  {emotion} percentage: {emotion_percentages[emotion][i]:.2f}%")
    
    # Return the results
    return bin_percentages, emotion_percentages


# Example usage
directories = [
    './ESD/Emotion Speech Dataset/0011/Angry', 
    './ESD/Emotion Speech Dataset/0011/Happy', 
    './ESD/Emotion Speech Dataset/0011/Neutral', 
    './ESD/Emotion Speech Dataset/0011/Sad', 
    './ESD/Emotion Speech Dataset/0011/Surprise', 
    './ESD/Emotion Speech Dataset/0012/Angry', 
    './ESD/Emotion Speech Dataset/0012/Happy', 
    './ESD/Emotion Speech Dataset/0012/Neutral', 
    './ESD/Emotion Speech Dataset/0012/Sad', 
    './ESD/Emotion Speech Dataset/0012/Surprise', 
    './ESD/Emotion Speech Dataset/0013/Angry', 
    './ESD/Emotion Speech Dataset/0013/Happy', 
    './ESD/Emotion Speech Dataset/0013/Neutral', 
    './ESD/Emotion Speech Dataset/0013/Sad', 
    './ESD/Emotion Speech Dataset/0013/Surprise', 
    './ESD/Emotion Speech Dataset/0014/Angry', 
    './ESD/Emotion Speech Dataset/0014/Happy', 
    './ESD/Emotion Speech Dataset/0014/Neutral', 
    './ESD/Emotion Speech Dataset/0014/Sad', 
    './ESD/Emotion Speech Dataset/0014/Surprise', 
    './ESD/Emotion Speech Dataset/0015/Angry', 
    './ESD/Emotion Speech Dataset/0015/Happy', 
    './ESD/Emotion Speech Dataset/0015/Neutral', 
    './ESD/Emotion Speech Dataset/0015/Sad', 
    './ESD/Emotion Speech Dataset/0015/Surprise', 
    './ESD/Emotion Speech Dataset/0016/Angry', 
    './ESD/Emotion Speech Dataset/0016/Happy', 
    './ESD/Emotion Speech Dataset/0016/Neutral', 
    './ESD/Emotion Speech Dataset/0016/Sad', 
    './ESD/Emotion Speech Dataset/0016/Surprise',
    './ESD/Emotion Speech Dataset/0017/Angry', 
    './ESD/Emotion Speech Dataset/0017/Happy', 
    './ESD/Emotion Speech Dataset/0017/Neutral', 
    './ESD/Emotion Speech Dataset/0017/Sad', 
    './ESD/Emotion Speech Dataset/0017/Surprise',  
    './ESD/Emotion Speech Dataset/0018/Angry', 
    './ESD/Emotion Speech Dataset/0018/Happy', 
    './ESD/Emotion Speech Dataset/0018/Neutral', 
    './ESD/Emotion Speech Dataset/0018/Sad', 
    './ESD/Emotion Speech Dataset/0018/Surprise', 
    './ESD/Emotion Speech Dataset/0019/Angry', 
    './ESD/Emotion Speech Dataset/0019/Happy', 
    './ESD/Emotion Speech Dataset/0019/Neutral', 
    './ESD/Emotion Speech Dataset/0019/Sad', 
    './ESD/Emotion Speech Dataset/0019/Surprise', 
    './ESD/Emotion Speech Dataset/0020/Angry', 
    './ESD/Emotion Speech Dataset/0020/Happy', 
    './ESD/Emotion Speech Dataset/0020/Neutral', 
    './ESD/Emotion Speech Dataset/0020/Sad', 
    './ESD/Emotion Speech Dataset/0020/Surprise'
]

directories2 = ["./CREMA-D/AudioWAV"]

emotions = ['Happy', 'Angry', 'Sad', 'Surprise', 'Neutral']
emotions2 = ['ANG', 'DIS', 'FEA', 'HAP', 'NEU', 'SAD']

# Call the function
calculate_duration_percentage_with_emotion_distribution(directories2, emotions2)