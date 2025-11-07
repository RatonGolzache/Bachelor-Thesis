import librosa
import os


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

def check_sampling_rate_uniformity(directories):
    for directory in directories:
        for file in os.listdir(directory):  # List files in the directory (no subdirectories)
            if file.endswith('.wav'):  # Check if it's a .wav file
                wav_path = os.path.join(directory, file)
                y, sr = librosa.load(wav_path, sr=None)
                print(f'{wav_path} - Sampling rate: {sr}')

check_sampling_rate_uniformity(directories2)

