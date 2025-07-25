from datetime import datetime
import librosa
import sys
import json
import numpy as np




def extract_pitch(y, sr):
    """Estimate the pitch of the audio signal using librosa piptrack."""
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_values = []
    magnitude_threshold = np.percentile(magnitudes, 75)  # filter out low magnitude pitches
    magnitudes = np.where(magnitudes < magnitude_threshold, 0, magnitudes)
    for i in range(pitches.shape[1]):
        # Get the pitch with the highest magnitude
        index = magnitudes[:, i].argmax()
        pitch = pitches[index, i]
        if 50 < pitch < 300: # typical human voice range
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
    speaking_rate = len(peaks) / duration_seconds * 60 # peaks per minute
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
        
        indicators = []
        
        if pitch < 85:
            indicators.append("Pitch is below normal, indicating possible fatigue or depression.")
        elif pitch > 250:
            indicators.append("Pitch is above normal, indicating possible excitement or anxiety.")

        if energy < 0.01:
            indicators.append("Energy level is low, indicating possible fatigue or lack of engagement.")
        elif energy > 0.1:
            indicators.append("Energy level is high, indicating possible excitement or engagement.")

        if speaking_rate < 120:
            indicators.append("Speaking rate is below normal, indicating possible boredom or fatigue.")
        elif speaking_rate > 220:
            indicators.append("Speaking rate is above normal, indicating possible excitement or anxiety.")

        analysis = []

        if pitch < 85 and energy < 0.01:
            analysis.append("The voice indicates a strong possibility of fatigue or depression due to low pitch and energy levels.")

        elif pitch > 250 and energy > 0.1:
            analysis.append("The voice indicates a strong possibility of excitement or anxiety due to high pitch and energy levels.")

        if speaking_rate > 220 and energy < 0.02:
            analysis.append("The voice indicates a strong possibility of anxiety or stress due to high speaking rate and low energy.")

        if not analysis:
            analysis.append("The voice does not indicate any significant health concerns based on the analyzed features.")

        # Prepare the result
        result = {
            "timestamp": datetime.now().isoformat(),   
            "file": audio_file,
            "features": [
                {
                    "name": "pitch",
                    "value": pitch,
                    "unit": "Hz"
                },
                {
                    "name": "energy",
                    "value": energy,
                    "unit": "normalized"
                },
                {
                    "name": "speaking_rate",
                    "value": speaking_rate,
                    "unit": "syllables per minute"
                }
            ],
            "health_indicators": indicators,
            "analysis": analysis
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