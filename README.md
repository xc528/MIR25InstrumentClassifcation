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


# Model 
# Evaluation 


