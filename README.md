# A Review of Emotional Speech Disentanglement Methods

## Overview
This project is a comprehensive review and implementation of methods for emotional speech disentanglement, inspired by the architecture proposed in *"Disentanglement of emotional style and speaker identity for expressive voice conversion"* by Zongyang Du et al., presented at Interspeech 2022. The codebase includes training scripts using [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D) and [Emotional Style Dataset (ESD)](https://github.com/HLTSingapore/Emotional-Speech-Data?tab=readme-ov-file) datasets, focused on disentangling emotional style and speaker identity in speech signals.

## Requirements
The project is developed with the following specific dependencies:
- Python: 3.8.20
- PyTorch: 2.0.1
- torchvision: 0.15.2
- tensorboard: 2.6.2.2
- nvidia-apex: 0.1
- soundfile: 0.10.3
- parallel_wavegan: 0.4.0
- mel-cepstral-distance: 0.0.4

All other dependencies are specified in the `requirements.txt` file, ensuring reproducibility of the environment.

## Dataset
- [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D)
- [Emotional Style Dataset (ESD)](https://github.com/HLTSingapore/Emotional-Speech-Data?tab=readme-ov-file)


