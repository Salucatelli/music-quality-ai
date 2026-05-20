import librosa
import numpy as np
from scipy import stats

def compute_features(audio_path):
    x, sr = librosa.load(audio_path, sr=22050, duration=30)
    
    def get_stats(feat):
        # ordem do fma: kurtosis, skew, max, mean, median, min, std
        return np.concatenate([
            stats.kurtosis(feat, axis=1),
            stats.skew(feat, axis=1),
            np.max(feat, axis=1),
            np.mean(feat, axis=1),
            np.median(feat, axis=1),
            np.min(feat, axis=1),
            np.std(feat, axis=1)
        ])

    # 1. Chroma STFT
    chroma = librosa.feature.chroma_stft(y=x, sr=sr, n_chroma=12)
    chroma_stats = get_stats(chroma) # 84

    # 2. Tonnetz (usa o chroma já calculado)
    tonnetz = librosa.feature.tonnetz(y=None, chroma=chroma)
    tonnetz_stats = get_stats(tonnetz) # 42

    # 3. MFCC
    mfcc = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=20)
    mfcc_stats = get_stats(mfcc) # 140

    # 4. Spectral Centroid
    spec_cent = librosa.feature.spectral_centroid(y=x, sr=sr)
    spec_cent_stats = get_stats(spec_cent) # 7

    # 5. Spectral Bandwidth
    spec_bw = librosa.feature.spectral_bandwidth(y=x, sr=sr)
    spec_bw_stats = get_stats(spec_bw) # 7

    # 6. Spectral Contrast (n_bands=6 gera 7 linhas)
    spec_cont = librosa.feature.spectral_contrast(y=x, sr=sr, n_bands=6)
    spec_cont_stats = get_stats(spec_cont) # 49

    # 7. Spectral Rolloff
    spec_roll = librosa.feature.spectral_rolloff(y=x, sr=sr)
    spec_roll_stats = get_stats(spec_roll) # 7

    # 8. RMS Energy
    rms = librosa.feature.rms(y=x)
    rms_stats = get_stats(rms) # 7

    # Concatenação na ordem do features.csv
    features_vector = np.concatenate([
        chroma_stats,     # 84
        tonnetz_stats,    # 42
        mfcc_stats,       # 140
        spec_cent_stats,  # 7
        spec_bw_stats,    # 7
        spec_cont_stats,  # 49
        spec_roll_stats,  # 7
        rms_stats         # 7
    ])
    
    return features_vector