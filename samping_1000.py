import os
import pandas as pd
import shutil
import glob

# Paths
base_dir = '/Users/ling/mir_datasets/medley_solos_db'
annotation_csv = os.path.join(base_dir, 'Medley-solos-DB_metadata.csv')
audio_dir = os.path.join(base_dir, 'audio')   # original audio folder
output_dir = os.path.join(base_dir, 'audio_1000_balanced')
output_csv = os.path.join(base_dir, 'audio_labels_1000_balanced.csv')

os.makedirs(output_dir, exist_ok=True)

# Read metadata CSV
meta_df = pd.read_csv(annotation_csv)
meta_df['uuid4'] = meta_df['uuid4'].astype(str).str.strip()  # remove potential spaces

# Collect all audio files
audio_files = glob.glob(os.path.join(audio_dir, '**', '*.wav'), recursive=True)
audio_files = [os.path.basename(f) for f in audio_files]  # keep only filenames
print(f"Found {len(audio_files)} audio files (including subdirectories).")

# Number of samples per class
num_per_class = 125
selected_rows = []

# Sample from each instrument_id
for inst_id in range(8):
    class_rows = meta_df[meta_df['instrument_id'] == inst_id]
    num_samples = min(num_per_class, len(class_rows))

    if num_samples > 0:
        sampled = class_rows.sample(n=num_samples, random_state=42)
        selected_rows.append(sampled)
        print(f"instrument_id {inst_id}: sampled {num_samples} rows")
    else:
        print(f"instrument_id {inst_id}: no available rows, skipped")

# Merge all sampled rows
if not selected_rows:
    raise ValueError("No rows sampled. Please check instrument_id values in the CSV.")

sample_df = pd.concat(selected_rows).reset_index(drop=True)
print(f"Total sampled rows: {len(sample_df)}")

# Copy matched audio files
copied_count_per_class = {i: 0 for i in range(8)}
copied_total = 0

for idx, row in sample_df.iterrows():
    uuid = row['uuid4']
    inst_id = row['instrument_id']

    matched_files = [f for f in audio_files if uuid in f]

    if matched_files:
        # Find the full path of the audio file
        src_file_path = glob.glob(os.path.join(audio_dir, '**', matched_files[0]), recursive=True)[0]
        dst_file = os.path.join(output_dir, matched_files[0])

        shutil.copy(src_file_path, dst_file)

        copied_count_per_class[inst_id] += 1
        copied_total += 1
    else:
        print(f"Audio file not found for uuid: {uuid}, instrument_id: {inst_id}")

# Print number of copied files per class
for inst_id, count in copied_count_per_class.items():
    print(f"instrument_id {inst_id}: copied {count} audio files")

print(f"Total copied audio files: {copied_total}")

# Save the sampled CSV
sample_df.to_csv(output_csv, index=False)
print(f"Saved label CSV to: {output_csv}")
