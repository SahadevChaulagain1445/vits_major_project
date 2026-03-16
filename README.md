# Nepali TTS Dataset Preprocessing Pipeline

## Overview
This project implements a complete preprocessing pipeline for Nepali Text-to-Speech (TTS) 
dataset preparation, following VITS architecture requirements.

## Project Structure
```
nepali_tts_project/
├── data/                 # Dataset files
├── preprocessing/        # Processing modules
├── tokenizers/          # Tokenizer files
├── configs/             # Configuration files
├── checkpoints/         # Model checkpoints
├── logs/                # Training logs
└── outputs/             # Generated samples
```

## Dataset Information
- Total Samples: 2740
- Language: Nepali (नेपाली)
- Format: WAV audio + Excel transcriptions

## Processing Steps
1. ✅ Project structure setup
2. ⏳ Dataset loading and verification
3. ⏳ Text normalization
4. ⏳ Grapheme-to-phoneme conversion
5. ⏳ Audio preprocessing
6. ⏳ Tokenizer creation
7. ⏳ Dataset splitting
8. ⏳ Metadata generation

## Usage
Follow the step-by-step preprocessing scripts in order.

## Requirements
- Python 3.8+
- pandas
- numpy
- librosa
- scipy
- pydub

## Author
PULCHOWK CAMPUS BEI(078) STUDENTS 

Sahadev Chaulagain(032)

Samyam Giri(035)

Sandip Acharya(036)

