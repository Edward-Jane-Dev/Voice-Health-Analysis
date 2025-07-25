from datetime import datetime
import librosa
import sys
import json
import numpy as np
import noisereduce as nr

pitch_thresholds = [85,255]  # typical human voice pitch range in Hz
energy_thresholds = [0.02, 0.1]  # normalized energy levels
speaking_rate_thresholds = [120, 220]  # syllables per minute


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
    """
    Calculate the energy of the audio signal. 
    Use RMS (Root Mean Square) to estimate energy.
    Filters out very low energy frames to avoid noise.
    """
    frame_length = 2048
    hop_length = 512
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    rms_nonzero = rms[rms > 0.01]
    if len(rms_nonzero) == 0:
        return 0.0
    return float(np.mean(rms_nonzero))

def extract_speaking_rate(y, sr):
    """
    Calculate the speaking rate in words per minute. 
    Uses 80th percentile of smoothed onset envelope to estimate the start of syllables. 
    Filters out closely spaced peaks.
    """
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_env_smooth = np.convolve(onset_env, np.ones(5)/5, mode='same')

    threshold = np.percentile(onset_env_smooth, 80)
    peaks = np.where(onset_env_smooth > threshold)[0]

    min_separation = int(0.08 * sr / 512)
    filtered_peaks = []
    last_peak = -min_separation
    for p in peaks:
        if p - last_peak >= min_separation:
            filtered_peaks.append(p)
            last_peak = p
    duration_seconds = len(y) / sr
    speaking_rate = len(filtered_peaks) / duration_seconds * 60 # peaks per minute
    return float(speaking_rate)

def analyze_voice(audio_file):
    try:
        # Load the audio file
        print(f"Loading audio file: {audio_file}")
        y, sr = librosa.load(audio_file, sr=None)
        print("Audio file loaded successfully.")

        # Reduce noise
        noise_clip = y[:int(sr * 0.5)]  # Use the first 0.5 seconds of the audio file to extract noise profile
        y = nr.reduce_noise(y, sr=sr, y_noise=noise_clip)

        # Extract features
        pitch = extract_pitch(y, sr)
        energy = extract_energy(y)
        speaking_rate = extract_speaking_rate(y, sr)
        
        indicators = []

        if pitch < pitch_thresholds[0]:
            indicators.append("Pitch is below normal, indicating possible fatigue or depression.")
        elif pitch > pitch_thresholds[1]:
            indicators.append("Pitch is above normal, indicating possible excitement or anxiety.")

        if energy < energy_thresholds[0]:
            indicators.append("Energy level is low, indicating possible fatigue.")
        elif energy > energy_thresholds[1]:
            indicators.append("Energy level is high, indicating possible excitement.")

        if speaking_rate < speaking_rate_thresholds[0]:
            indicators.append("Speaking rate is below normal, indicating possible boredom or fatigue.")
        elif speaking_rate > speaking_rate_thresholds[1]:
            indicators.append("Speaking rate is above normal, indicating possible excitement or anxiety.")

        analysis = []

        # Analyze the features and provide health indicators
        if pitch < pitch_thresholds[0] and energy < energy_thresholds[0]:
            analysis.append("The voice indicates a strong possibility of fatigue or depression due to low pitch and energy levels.")

        elif pitch > pitch_thresholds[1] and energy > energy_thresholds[1]:
            analysis.append("The voice indicates a strong possibility of excitement or anxiety due to high pitch and energy levels.")

        elif speaking_rate > speaking_rate_thresholds[1] and energy < energy_thresholds[0]:
            analysis.append("The voice indicates a strong possibility of anxiety or stress due to high speaking rate and low energy.")
        
        if not analysis: # If no specific analysis was made, check individual features
            if pitch < pitch_thresholds[0]:
                analysis.append("The voice indicates possible fatigue or depression due to low pitch.")
            if pitch > pitch_thresholds[1]:
                analysis.append("The voice indicates possible excitement or anxiety due to high pitch.")
            if energy < energy_thresholds[0]:
                analysis.append("The voice indicates possible fatigue due to low energy.")
            if energy > energy_thresholds[1]:
                analysis.append("The voice indicates possible excitement due to high energy levels.")
            if speaking_rate < speaking_rate_thresholds[0]:
                analysis.append("The voice indicates possible boredom or fatigue due to low speaking rate.")
            if speaking_rate > speaking_rate_thresholds[1]:
                analysis.append("The voice indicates possible excitement or anxiety due to high speaking rate.")
            
        if not analysis: # If there is still no analysis, assume normal
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