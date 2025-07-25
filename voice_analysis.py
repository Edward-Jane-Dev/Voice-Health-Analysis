from datetime import datetime
import librosa
import sys
import json
import numpy as np




def extract_pitch(y, sr):
    """Estimate the pitch of the audio signal using librosa piptrack."""
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_values = []
    for i in range(pitches.shape[1]):
        # Get the pitch with the highest magnitude
        index = magnitudes[:, i].argmax()
        pitch = pitches[index, i]
        if 50 < pitch < 400: # typical human voice range
            pitch_values.append(pitch)
    return float(np.median(pitch_values)) if pitch_values else None

def extract_energy(y):
    """Calculate the energy of the audio signal. Use the RMS energy formula."""
    energy = np.sum(y**2) / len(y)
    return float(energy)

def extract_speaking_rate(y, sr):
    """Calculate the speaking rate in words per minute. Uses onset detection to estimate the start of syllables."""
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    peaks = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, units='time')
    duration_seconds = len(y) / sr
    speaking_rate = len(peaks) / duration_seconds * 60   # peaks per minute
    return float(speaking_rate)

def analyze_voice(audio_file):
    try:
        # Load the audio file
        print(f"Loading audio file: {audio_file}")
        y, sr = librosa.load(audio_file, sr=None)
        print("Audio file loaded successfully.")

        # Extract features
        pitch = extract_pitch(y, sr)
        energy = extract_energy(y)
        speaking_rate = extract_speaking_rate(y, sr)
        
        # Prepare the result
        result = {
            "timestamp": datetime.now().isoformat(),   
            "file": audio_file,
            "features": {
                "pitch": pitch,
                "energy": energy,
                "speaking_rate": speaking_rate
            }
        }
        
        return result
    
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python voice_analysis.py <audio_file>")
        sys.exit(1)
    audio_file = sys.argv[1]
    result = analyze_voice(audio_file)
    print(json.dumps(result, indent=2))