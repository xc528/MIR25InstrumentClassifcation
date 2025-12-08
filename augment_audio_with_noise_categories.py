"""
Audio Augmentation Script with Noise Categories

This script augments 1000 original audio files by mixing them with noise from 10 categories,
generating 10,000 augmented audio files (1000 × 10 categories).
"""

import os
import csv
import random
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torchaudio
from tqdm import tqdm
import numpy as np

# Configuration
SAMPLE_RATE = 44100
NOISE_AMPLITUDE_MIN = 0.6
NOISE_AMPLITUDE_MAX = 1.4
PITCH_SHIFT_MIN = -2  # semitones
PITCH_SHIFT_MAX = 2   # semitones
SNR_MIN = 8   # dB
SNR_MAX = 20  # dB

# Paths
AUDIO_DIR = Path("audio_1000_balanced")
NOISES_DIR = Path("noises")
OUTPUT_AUDIO_DIR = Path("augmented_audio")
CSV_INPUT = "audio_labels_1000_balanced.csv"
CSV_OUTPUT = "augmented_audio_labels_10000.csv"

# Noise categories (auto-detected from folder names)
NOISE_CATEGORIES = [
    "ambience", "applause", "bird", "crowd", "fan",
    "microphone", "rain", "street", "talking", "white_noise"
]


def find_audio_file(uuid4: str, audio_dir: Path) -> Optional[Path]:
    """
    Find audio file by UUID4.
    
    Args:
        uuid4: UUID4 string from CSV
        audio_dir: Directory containing audio files
        
    Returns:
        Path to audio file or None if not found
    """
    pattern = f"*{uuid4}*.wav"
    matches = [f for f in audio_dir.glob(pattern) if not f.name.startswith("._")]
    if matches:
        return matches[0]
    return None


def load_audio(file_path: Path, target_sr: int = SAMPLE_RATE) -> torch.Tensor:
    """
    Load audio file, convert to mono, and resample to target sample rate.
    
    Args:
        file_path: Path to audio file
        target_sr: Target sample rate
        
    Returns:
        Audio tensor of shape (1, samples) - mono channel
    """
    with torch.no_grad():
        waveform, sr = torchaudio.load(str(file_path))
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if needed
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            waveform = resampler(waveform)
        
        # Ensure no gradients
        waveform = waveform.detach()
    
    return waveform


def apply_pitch_shift(waveform: torch.Tensor, n_steps: float, sample_rate: int = SAMPLE_RATE) -> torch.Tensor:
    """
    Apply pitch shift to audio waveform.
    
    Args:
        waveform: Audio tensor of shape (1, samples)
        n_steps: Number of semitones to shift (can be negative)
        sample_rate: Sample rate of the audio
        
    Returns:
        Pitch-shifted audio tensor
    """
    if n_steps == 0:
        return waveform
    
    # Use torchaudio's pitch shift
    pitch_shift = torchaudio.transforms.PitchShift(
        sample_rate=sample_rate,
        n_steps=n_steps
    )
    return pitch_shift(waveform)


def apply_amplitude_gain(waveform: torch.Tensor, gain: float) -> torch.Tensor:
    """
    Apply amplitude gain to audio waveform.
    
    Args:
        waveform: Audio tensor
        gain: Gain multiplier
        
    Returns:
        Scaled audio tensor
    """
    return waveform * gain


def match_length(noise: torch.Tensor, target_length: int) -> torch.Tensor:
    """
    Match noise length to target length by looping or trimming.
    
    Args:
        noise: Noise tensor of shape (1, samples)
        target_length: Target length in samples
        
    Returns:
        Noise tensor with matching length
    """
    noise_length = noise.shape[1]
    
    if noise_length < target_length:
        # Loop the noise
        num_repeats = (target_length // noise_length) + 1
        noise = noise.repeat(1, num_repeats)
        noise = noise[:, :target_length]
    elif noise_length > target_length:
        # Trim the noise
        noise = noise[:, :target_length]
    
    return noise


def mix_with_snr(clean: torch.Tensor, noise: torch.Tensor, snr_db: float) -> torch.Tensor:
    """
    Mix clean audio with noise at specified SNR.
    
    Args:
        clean: Clean audio tensor of shape (1, samples)
        noise: Noise tensor of shape (1, samples)
        snr_db: Signal-to-noise ratio in dB
        
    Returns:
        Mixed audio tensor
    """
    # Ensure inputs don't require grad
    clean = clean.detach() if clean.requires_grad else clean
    noise = noise.detach() if noise.requires_grad else noise
    
    # Calculate RMS
    clean_rms = torch.sqrt(torch.mean(clean ** 2))
    noise_rms = torch.sqrt(torch.mean(noise ** 2))
    
    # Avoid division by zero
    if noise_rms == 0:
        return clean
    
    # Calculate scaling factor for noise
    # SNR = 20 * log10(clean_rms / noise_scaled_rms)
    # noise_scaled_rms = clean_rms / 10^(SNR/20)
    target_noise_rms = clean_rms / (10 ** (snr_db / 20.0))
    noise_scale = target_noise_rms / noise_rms
    
    # Scale noise
    noise_scaled = noise * noise_scale
    
    # Mix
    mixed = clean + noise_scaled
    
    # Normalize if clipping occurs
    max_val = torch.max(torch.abs(mixed))
    if max_val > 1.0:
        mixed = mixed / max_val
    
    # Ensure output doesn't require grad
    return mixed.detach()


def load_noise_files(noises_dir: Path) -> Dict[str, List[Path]]:
    """
    Load all noise files from each category folder.
    
    Args:
        noises_dir: Root directory containing noise category folders
        
    Returns:
        Dictionary mapping category names to lists of noise file paths
    """
    noise_files = {}
    
    for category in NOISE_CATEGORIES:
        category_dir = noises_dir / category
        if not category_dir.exists():
            print(f"Warning: Category folder '{category}' not found, skipping.")
            continue
        
        # Find all audio files in the category folder
        files = sorted(list(category_dir.glob("*.mp3")) + list(category_dir.glob("*.wav")))
        if len(files) == 0:
            print(f"Warning: No audio files found in '{category}', skipping.")
            continue
        
        noise_files[category] = files
        print(f"Loaded {len(files)} noise files from '{category}'")
    
    return noise_files


def augment_audio(
    clean_audio: torch.Tensor,
    noise_file: Path,
    snr_db: float,
    amplitude_gain: float,
    pitch_shift: float
) -> torch.Tensor:
    """
    Augment clean audio with noise, applying transformations.
    
    Args:
        clean_audio: Clean audio tensor of shape (1, samples)
        noise_file: Path to noise file
        snr_db: Signal-to-noise ratio in dB
        amplitude_gain: Amplitude gain for noise (0.6-1.4)
        pitch_shift: Pitch shift in semitones (-2 to +2)
        
    Returns:
        Augmented audio tensor
    """
    # Ensure clean audio doesn't require grad
    clean_audio = clean_audio.detach() if clean_audio.requires_grad else clean_audio
    
    # Load noise
    noise = load_audio(noise_file)
    
    # Apply transformations to noise
    noise = apply_amplitude_gain(noise, amplitude_gain)
    noise = apply_pitch_shift(noise, pitch_shift)
    
    # Match length
    target_length = clean_audio.shape[1]
    noise = match_length(noise, target_length)
    
    # Mix with SNR
    augmented = mix_with_snr(clean_audio, noise, snr_db)
    
    # Ensure final output doesn't require grad
    return augmented.detach()


def main():
    """Main function to perform audio augmentation."""
    
    # Set random seed for reproducibility
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    OUTPUT_AUDIO_DIR.mkdir(exist_ok=True)
    print(f"Output directory: {OUTPUT_AUDIO_DIR}")
    
    # Load noise files
    print("\nLoading noise files...")
    noise_files = load_noise_files(NOISES_DIR)
    
    if len(noise_files) == 0:
        print("Error: No noise files found!")
        return
    
    print(f"\nFound {len(noise_files)} noise categories")
    
    # Load original CSV
    print(f"\nLoading CSV: {CSV_INPUT}")
    original_rows = []
    with open(CSV_INPUT, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            original_rows.append(row)
    
    print(f"Loaded {len(original_rows)} original audio files")
    
    # Prepare output CSV
    output_rows = []
    fieldnames = list(original_rows[0].keys()) + ['noise_category']
    
    # Load existing CSV if it exists to resume
    existing_entries = {}
    if Path(CSV_OUTPUT).exists():
        print(f"\nFound existing CSV: {CSV_OUTPUT}")
        with open(CSV_OUTPUT, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Create a key from original metadata + noise category
                key = (row['uuid4'], row['noise_category'])
                existing_entries[key] = row
        print(f"Loaded {len(existing_entries)} existing entries from CSV")
    
    # Track which files already exist
    existing_files = set()
    if OUTPUT_AUDIO_DIR.exists():
        for f in OUTPUT_AUDIO_DIR.glob("*.wav"):
            existing_files.add(f.name)
        print(f"Found {len(existing_files)} existing augmented files in {OUTPUT_AUDIO_DIR}")
    
    # Process each original audio file
    print("\nStarting augmentation...")
    failed_files = []
    files_processed = 0
    augmented_files_count = 0
    csv_save_interval = 1000  # Save CSV every 1000 augmented files
    
    for original_row in tqdm(original_rows, desc="Processing audio files"):
        uuid4 = original_row['uuid4']
        
        # Find audio file
        audio_file = find_audio_file(uuid4, AUDIO_DIR)
        if audio_file is None:
            print(f"\nWarning: Audio file not found for UUID {uuid4}, skipping.")
            failed_files.append(uuid4)
            continue
        
        # Load clean audio
        try:
            clean_audio = load_audio(audio_file)
        except Exception as e:
            print(f"\nError loading audio file {audio_file}: {e}")
            failed_files.append(uuid4)
            continue
        
        # Get original filename without extension for output naming
        original_filename = audio_file.stem
        
        # Augment with each noise category
        for category in NOISE_CATEGORIES:
            if category not in noise_files:
                continue
            
            # Check if this file already exists
            output_filename = f"{original_filename}_{category}.wav"
            output_path = OUTPUT_AUDIO_DIR / output_filename
            entry_key = (uuid4, category)
            
            # Skip if file exists and entry is in CSV
            if output_filename in existing_files and entry_key in existing_entries:
                # Add existing entry to output_rows to maintain complete CSV
                output_rows.append(existing_entries[entry_key])
                augmented_files_count += 1  # Count skipped files too
                continue
            
            # Randomly select one noise file from this category
            noise_file = random.choice(noise_files[category])
            
            # Generate random parameters
            amplitude_gain = random.uniform(NOISE_AMPLITUDE_MIN, NOISE_AMPLITUDE_MAX)
            pitch_shift = random.uniform(PITCH_SHIFT_MIN, PITCH_SHIFT_MAX)
            snr_db = random.uniform(SNR_MIN, SNR_MAX)
            
            # Augment audio
            try:
                augmented_audio = augment_audio(
                    clean_audio,
                    noise_file,
                    snr_db,
                    amplitude_gain,
                    pitch_shift
                )
                
                # Ensure tensor doesn't require grad for saving (with no_grad context)
                with torch.no_grad():
                    augmented_audio = augmented_audio.detach().clone()
                
                torchaudio.save(
                    str(output_path),
                    augmented_audio,
                    SAMPLE_RATE
                )
                
                # Add row to output CSV
                new_row = original_row.copy()
                new_row['noise_category'] = category
                output_rows.append(new_row)
                
                # Update existing files set
                existing_files.add(output_filename)
                augmented_files_count += 1
                
            except Exception as e:
                print(f"\nError augmenting {audio_file} with {category}: {e}")
                continue
        
        files_processed += 1
        
        # Save CSV incrementally every 1000 augmented files
        if augmented_files_count > 0 and augmented_files_count % csv_save_interval == 0:
            print(f"\nSaving incremental CSV ({augmented_files_count} augmented files, {files_processed} original files processed)...")
            # Merge existing + new entries for incremental save
            incremental_rows = []
            processed_keys_incremental = set()
            
            # Add all processed entries so far
            for row in output_rows:
                key = (row['uuid4'], row['noise_category'])
                processed_keys_incremental.add(key)
                incremental_rows.append(row)
            
            # Add existing entries that haven't been reprocessed
            for key, row in existing_entries.items():
                if key not in processed_keys_incremental:
                    incremental_rows.append(row)
            
            # Sort for consistency
            incremental_rows.sort(key=lambda x: (x['uuid4'], x['noise_category']))
            
            with open(CSV_OUTPUT, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(incremental_rows)
            print(f"Saved {len(incremental_rows)} entries to {CSV_OUTPUT}")
    
    # Write final output CSV (merge existing + new entries)
    print(f"\nWriting final output CSV: {CSV_OUTPUT}")
    
    # Merge existing entries that weren't reprocessed
    all_rows = []
    processed_keys = set()
    
    # Add all processed entries
    for row in output_rows:
        key = (row['uuid4'], row['noise_category'])
        processed_keys.add(key)
        all_rows.append(row)
    
    # Add existing entries that weren't reprocessed
    for key, row in existing_entries.items():
        if key not in processed_keys:
            all_rows.append(row)
    
    # Sort by uuid4 and noise_category for consistency
    all_rows.sort(key=lambda x: (x['uuid4'], x['noise_category']))
    
    with open(CSV_OUTPUT, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    
    print(f"Final CSV contains {len(all_rows)} entries (expected: {len(original_rows) * len(NOISE_CATEGORIES)})")
    
    # Print summary
    print("\n" + "="*60)
    print("AUGMENTATION SUMMARY")
    print("="*60)
    print(f"Number of noise categories found: {len(noise_files)}")
    print(f"Number of original audio files processed: {files_processed}")
    print(f"Number of new augmented files generated in this run: {len([r for r in output_rows if (r['uuid4'], r['noise_category']) not in existing_entries])}")
    print(f"Total augmented files (existing + new): {len(all_rows)}")
    print(f"Expected total augmented files: {len(original_rows) * len(NOISE_CATEGORIES)}")
    
    if failed_files:
        print(f"\nWarning: {len(failed_files)} files failed to process")
    
    print(f"\nOutput audio directory: {OUTPUT_AUDIO_DIR}")
    print(f"Output CSV: {CSV_OUTPUT}")
    print("="*60)


if __name__ == "__main__":
    main()

