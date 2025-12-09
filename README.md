# MIR25InstrumentClassifcation
## How do state of the art methods perform in live settings?
This repository contains the code and experimentation for a Music Information Retrieval (MIR) project analyzing AudioMAE and openL3 embeddings under different acoustic conditions (dry, reverb, distortion). The project investigates k-means clustering conditions of the two models using audio processed with these different conditions.

# Audio Data Processing 
## samping_1000.py
This script is designed to sample a balanced subset of audio files from the Medley-Solos-DB dataset and generate a corresponding label CSV file. It extracts a specified number of samples per instrument.
### Features:
Read all WAV files from the original audio folder.
Sample a fixed number of audio files per instrument_id from the metadata CSV.
Copy sampled audio files to a new output folder.
Save the sampled label CSV file.
### Parameters:
base_dir: str
Path to the root folder of the Medley-Solos-DB dataset. Should contain the audio folder and metadata CSV.
annotation_csv: str
Path to the metadata CSV file (e.g., Medley-solos-DB_metadata.csv).
audio_dir: str
Path to the folder containing the original audio files.
output_dir: str
Path to the folder where the sampled audio files will be copied.
output_csv: str
Path to save the CSV file containing labels for the sampled audio.
num_per_class: int
Number of audio samples to select per instrument_id. Default is 125.
instrument_id range: int
The IDs of instruments to sample (0–7 in this script, corresponding to 8 instruments).

## apply_reverb.py
This script applies multiple reverb effects to a set of audio files from the Medley-Solos-DB dataset. Each reverb is defined by an impulse response (IR) WAV file. For each reverb applied, the script generates a new folder of processed audio and a corresponding label CSV.
### Features:
Load original audio files from a balanced subset (e.g., 1000 samples).
Apply multiple reverb effects using convolution with IR WAV files.
Normalize audio after convolution to prevent clipping.
Save processed audio files in separate folders for each reverb.
Generate a corresponding CSV file containing reverb labels for each audio file.
### Parameters:
ori_audio_path: str
Path to the folder containing the original audio files.
csv_path: str
Path to the CSV file containing labels for the original audio.
ir_folder: str
Path to the folder containing IR WAV files (Reverb1.wav ~ Reverb10.wav).
ir_wavs: list of str
List of IR filenames to apply. Default: ['Reverb1.wav', ..., 'Reverb10.wav'].
output_audio_path: str
Path to save the processed audio files. Automatically creates subfolders for each reverb.
df_new['reverb_label']: str
Column added to the CSV to indicate which reverb effect was applied.

# Data Augmentation - Noise

This section augments 1000 original audio files by mixing them with noise from 10 categories, generating **10,000 augmented audio files** (1000 × 10 categories).

## What It Does

The notebook `augment_audio_with_noise/augment_audio_with_noise_categories.ipynb` performs audio data augmentation using files in the `augment_audio_with_noise/` folder. It:
- Loads original audio files and mixes them with noise from 10 categories
- Applies random transformations (amplitude gain, pitch shift)
- Mixes at controlled SNR levels (8-20 dB) to ensure clean audio dominates
- Generates augmented files with metadata in CSV format

**Noise Categories**: ambience, applause, bird, crowd, fan, microphone, rain, street, talking, white_noise

## Files Used

**Input Files**:
- `audio_1000_balanced/` - Directory containing 1000 original .wav audio files
- `audio_labels_1000_balanced.csv` - CSV with metadata columns: `subset`, `instrument`, `instrument_id`, `song_id`, `uuid4`
- `augment_audio_with_noise/noises/` - Directory with 10 category subfolders, each containing 5 noise files (.mp3 or .wav)

**Notebook**:
- `augment_audio_with_noise/augment_audio_with_noise_categories.ipynb` - Main augmentation notebook

**Output Files**:
- `augment_audio_with_noise/augmented_audio/` - Directory containing 10,000 augmented .wav files
- `augment_audio_with_noise/augmented_audio_noise_10000.csv` - CSV with metadata for all augmented files (original columns + `noise_category`)

**Demo Files**:
- `augment_audio_with_noise/demo/` - Contains example augmented files: 10 augmented versions (one per noise category) of a single original audio file, plus the original file for comparison

## How to Run

1. **Open the notebook in Google Colab**:
   - Upload `augment_audio_with_noise/augment_audio_with_noise_categories.ipynb`
   - Upload required data files (see Files Used section)

2. **Update paths in Section 2 (Configuration)**:
   ```python
   AUDIO_DIR = Path("audio_1000_balanced")  # Adjust path based on Colab upload location
   NOISES_DIR = Path("noises")
   OUTPUT_AUDIO_DIR = Path("augmented_audio")
   CSV_INPUT = "audio_labels_1000_balanced.csv"
   CSV_OUTPUT = "augmented_audio_noise_10000.csv"
   ```

3. **Run all cells**:
   - Execute cells sequentially from top to bottom
   - Or use "Run All" to execute the entire notebook
    
## Output

- **Augmented Files**: `<original_filename>_<noise_category>.wav` in `augment_audio_with_noise/augmented_audio/`
- **CSV Metadata**: Contains all original columns plus `noise_category` column
- **Total**: 10,000 augmented files (1000 originals × 10 noise categories)

# Model 
# Evaluation 
To evaluate how well the two models perform with the processed audio, we ran the embedding with K-means Clustering, and compared the clustering result with their ground truths(instrument labels).

## Files Used

**Input Files**:
- `audio_labels_1000_balanced.csv` - CSV with metadata columns: `subset`, `instrument`, `instrument_id`, `song_id`, `uuid4`
-  folders with embeddings - each folder should be a complete set of 1000 .npy files from a model of either dry or processed data

**Notebook**:
- `Evaluation/Eval.ipynb` - Main evaluation notebook, contains functions to load, analyze the embedings and produce graphs output

**Demo Files**:
- `Evaluation/graphs/` - Contains example graphs comparing clusters colord by k-means clustering vs by instrument and their confusion Matrix produced by the openL3 model on audio processed with noise. 

## How to Run

1. **Open the notebook**:
   - Upload `augment_audio_with_noise/augment_audio_with_noise_categories.ipynb`
   - Upload required data files

2. **Update paths in cell2**:
   ```python
   #replace path name with your own
   #csv_path = "/Users/xinranchen/Downloads/audio_labels.csv"
   openL3_path = "/Users/xinranchen/Desktop/MIRGroupAssignment/noise/openl3_mel256_music_512/"
   audioMAE_path = "/Users/xinranchen/Desktop/MIRGroupAssignment/noise/audiomae_base_768/"
   save_path = "/Users/xinranchen/Desktop/MIRGroupAssignment/graph/noise"
   pathFolder = ["ambience", "applause", "bird", "crowd", "fan", "microphone", "rain", "street", "talking", "white_noise" ]
   ```
3.**Modify Cells with Folders and Model Name**
   ```python
   #example:
   X, instrument_list, id_list = loadFolder(f"{audioMAE_path}{"applause"}")
   k_means_clustering(X, instrument_list, "applause")
   ```
# Conclusion
