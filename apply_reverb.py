import os
import numpy as np
import pandas as pd
import librosa
import soundfile as sf

def add_reverb(input_audio_name, output_audio_name, ir):
    audio, sr = librosa.load(input_audio_name, sr=None)
    reverb_audio = np.convolve(audio, ir, mode='full')
    reverb_audio = reverb_audio / np.max(np.abs(reverb_audio))  # Normalize the audio
    sf.write(output_audio_name, reverb_audio, sr)

if __name__ == '__main__':
    # original audio path 
    ori_audio_path = '/Users/ling/mir_datasets/medley_solos_db/audio_1000_balanced'
    
    # original CSV path
    csv_path = '/Users/ling/mir_datasets/medley_solos_db/audio_labels_1000_balanced.csv'
    df = pd.read_csv(csv_path)

    # IR Reverb1-Reverb10
    ir_wavs = [f'Reverb{i}.wav' for i in range(1, 11)]
    ir_folder = '/Users/ling/mir_datasets/medley_solos_db/ir_wavs'

    for ir_name in ir_wavs:
        ir_path = os.path.join(ir_folder, ir_name)
        if not os.path.exists(ir_path):
            raise FileNotFoundError(f"IR file not found: {ir_path}")
        ir, sr_ir = librosa.load(ir_path, sr=None)

        # Output directory for processed audio
        output_audio_path = ori_audio_path + f'_{ir_name.split(".")[0]}'
        os.makedirs(output_audio_path, exist_ok=True)

        # List all audio files in the original directory
        audio_filenames = [f for f in os.listdir(ori_audio_path)
                           if f.endswith('.wav') and not f.startswith('._')]

        # Create a new DataFrame containing only labels for files that exist
        df_new = df[df['uuid4'].apply(lambda x: any(x in f for f in audio_filenames))].copy()
        df_new['reverb_label'] = ir_name.split('.')[0]  # 'Reverb5' ~ 'Reverb10'

        # Apply reverb to each audio file
        for filename in audio_filenames:
            input_audio_name = os.path.join(ori_audio_path, filename)
            output_audio_name = os.path.join(output_audio_path, filename)
            add_reverb(input_audio_name, output_audio_name, ir)

        # save updated CSV file
        new_csv_path = os.path.join('/Users/ling/mir_datasets/medley_solos_db', 
                                    f'audio_labels_1000_balanced_{ir_name.split(".")[0]}.csv')
        df_new.to_csv(new_csv_path, index=False)

        print(f"{ir_name} applied. Audio saved to {output_audio_path}, CSV saved to {new_csv_path}")
