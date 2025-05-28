"""
Audio Analysis Module for Streamlit Song Creation System

This module handles the analysis of audio features using librosa.
"""

import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tempfile
import logging
from typing import Dict, Any, Optional, List, Tuple
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioAnalyzer:
    """Class for analyzing audio features."""
    
    def __init__(self, output_dir: str = None):
        """
        Initialize the audio analyzer.
        
        Args:
            output_dir: Directory to save analysis results and visualizations
        """
        self.output_dir = output_dir or tempfile.gettempdir()
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"Audio analyzer initialized with output directory: {self.output_dir}")
    
    def analyze_audio(self, audio_file: str, progress_callback=None) -> Dict[str, Any]:
        """
        Analyze audio features from a file.
        
        Args:
            audio_file: Path to the audio file
            progress_callback: Optional callback function for progress updates
            
        Returns:
            Dictionary containing analysis results
        """
        logger.info(f"Analyzing audio file: {audio_file}")
        
        # Load audio file
        try:
            y, sr = librosa.load(audio_file, sr=None)
            duration = librosa.get_duration(y=y, sr=sr)
            
            if progress_callback:
                progress_callback(0.1)  # 10% progress after loading
            
            # Initialize results dictionary
            results = {
                'file_path': audio_file,
                'duration': duration,
                'sample_rate': sr,
                'rhythm': {},
                'harmony': {},
                'timbre': {},
                'structure': {},
                'visualizations': {}
            }
            
            # Analyze rhythm features
            results['rhythm'] = self._analyze_rhythm(y, sr)
            if progress_callback:
                progress_callback(0.3)  # 30% progress after rhythm analysis
            
            # Analyze harmonic features
            results['harmony'] = self._analyze_harmony(y, sr)
            if progress_callback:
                progress_callback(0.5)  # 50% progress after harmony analysis
            
            # Analyze timbre features
            results['timbre'] = self._analyze_timbre(y, sr)
            if progress_callback:
                progress_callback(0.7)  # 70% progress after timbre analysis
            
            # Analyze structure
            results['structure'] = self._analyze_structure(y, sr)
            if progress_callback:
                progress_callback(0.9)  # 90% progress after structure analysis
            
            # Generate visualizations
            results['visualizations'] = self._generate_visualizations(y, sr, audio_file)
            if progress_callback:
                progress_callback(1.0)  # 100% progress after visualizations
            
            # Save results to JSON file
            output_json = os.path.join(self.output_dir, os.path.basename(audio_file) + '_analysis.json')
            with open(output_json, 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                serializable_results = self._make_json_serializable(results)
                json.dump(serializable_results, f, indent=2)
            
            logger.info(f"Analysis completed and saved to: {output_json}")
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing audio: {str(e)}")
            raise
    
    def _analyze_rhythm(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Analyze rhythm features."""
        logger.info("Analyzing rhythm features")
        
        # Extract tempo and beat frames
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        
        # Convert beat frames to time
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        
        # Calculate beat strength
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        pulse = librosa.beat.plp(onset_envelope=onset_env, sr=sr)
        beat_strength = np.mean(pulse[beat_frames])
        
        # Calculate rhythm regularity (lower variance means more regular)
        if len(beat_times) > 1:
            beat_intervals = np.diff(beat_times)
            rhythm_regularity = 1.0 / (1.0 + np.std(beat_intervals) / np.mean(beat_intervals))
        else:
            rhythm_regularity = 0.0
        
        return {
            'tempo': float(tempo),
            'beat_times': beat_times.tolist(),
            'beat_strength': float(beat_strength),
            'rhythm_regularity': float(rhythm_regularity)
        }
    
    def _analyze_harmony(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Analyze harmonic features."""
        logger.info("Analyzing harmonic features")
        
        # Compute chromagram
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        
        # Estimate key
        key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        mode_names = ['Major', 'Minor']
        
        # Sum over time to get overall key profile
        chroma_sum = np.sum(chroma, axis=1)
        key_index = np.argmax(chroma_sum)
        key_strength = chroma_sum[key_index] / np.sum(chroma_sum)
        
        # Simple mode detection (major/minor)
        major_profile = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])  # C major
        minor_profile = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0])  # C minor
        
        # Rotate profiles to match detected key
        major_profile = np.roll(major_profile, key_index)
        minor_profile = np.roll(minor_profile, key_index)
        
        # Correlate with chroma
        major_correlation = np.corrcoef(chroma_sum, major_profile)[0, 1]
        minor_correlation = np.corrcoef(chroma_sum, minor_profile)[0, 1]
        
        # Determine mode
        mode_index = 0 if major_correlation > minor_correlation else 1
        mode_confidence = max(major_correlation, minor_correlation)
        
        # Estimate chord complexity
        # Higher standard deviation in chroma indicates more complex harmony
        chord_complexity = float(np.std(chroma_sum))
        
        # Normalize to 0-1 range
        chord_complexity = min(1.0, chord_complexity / 0.5)
        
        # Estimate harmonic novelty
        # Higher frame-to-frame changes indicate more harmonic novelty
        harmonic_novelty = float(np.mean(np.diff(chroma, axis=1)**2))
        
        # Normalize to 0-1 range
        harmonic_novelty = min(1.0, harmonic_novelty / 0.1)
        
        return {
            'key': key_names[key_index],
            'mode': mode_names[mode_index],
            'key_strength': float(key_strength),
            'mode_confidence': float(mode_confidence),
            'chord_complexity': float(chord_complexity),
            'harmonic_novelty': float(harmonic_novelty),
            'chroma': chroma.tolist()
        }
    
    def _analyze_timbre(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Analyze timbre features."""
        logger.info("Analyzing timbre features")
        
        # Compute MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        # Compute spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        
        # Compute zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        
        # Simple instrument detection (very basic approximation)
        # In a real implementation, this would use a trained classifier
        instruments = {
            'Guitar': 0.0,
            'Piano': 0.0,
            'Drums': 0.0,
            'Bass': 0.0,
            'Vocals': 0.0,
            'Strings': 0.0,
            'Synth': 0.0
        }
        
        # Very simplified heuristics for instrument detection
        # High spectral centroid often indicates presence of cymbals/drums
        if np.mean(spectral_centroid) > 3000:
            instruments['Drums'] = min(1.0, np.mean(spectral_centroid) / 5000)
        
        # High zero crossing rate often indicates vocals or strings
        if np.mean(zcr) > 0.05:
            instruments['Vocals'] = min(1.0, np.mean(zcr) / 0.1)
        
        # Low spectral centroid often indicates bass
        if np.mean(spectral_centroid) < 1000:
            instruments['Bass'] = min(1.0, 1.0 - np.mean(spectral_centroid) / 1000)
        
        # Medium spectral centroid and low contrast often indicates piano
        if 1000 < np.mean(spectral_centroid) < 2000 and np.mean(spectral_contrast) < 10:
            instruments['Piano'] = min(1.0, np.mean(spectral_contrast) / 10)
        
        # Medium spectral centroid and high contrast often indicates guitar
        if 1000 < np.mean(spectral_centroid) < 3000 and np.mean(spectral_contrast) > 10:
            instruments['Guitar'] = min(1.0, np.mean(spectral_contrast) / 20)
        
        # High spectral bandwidth often indicates synth
        if np.mean(spectral_bandwidth) > 3000:
            instruments['Synth'] = min(1.0, np.mean(spectral_bandwidth) / 5000)
        
        return {
            'mfccs_mean': np.mean(mfccs, axis=1).tolist(),
            'mfccs_std': np.std(mfccs, axis=1).tolist(),
            'spectral_centroid_mean': float(np.mean(spectral_centroid)),
            'spectral_bandwidth_mean': float(np.mean(spectral_bandwidth)),
            'spectral_contrast_mean': np.mean(spectral_contrast, axis=1).tolist(),
            'spectral_rolloff_mean': float(np.mean(spectral_rolloff)),
            'zero_crossing_rate_mean': float(np.mean(zcr)),
            'instruments': instruments
        }
    
    def _analyze_structure(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Analyze song structure."""
        logger.info("Analyzing song structure")
        
        # Compute MFCC features for structure analysis
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        # Normalize MFCCs
        mfccs = librosa.util.normalize(mfccs, axis=1)
        
        # Compute self-similarity matrix
        S = librosa.segment.recurrence_matrix(mfccs, mode='affinity', sym=True)
        
        # Use spectral clustering to identify segments
        n_segments = min(8, int(librosa.get_duration(y=y, sr=sr) / 30) + 1)  # Estimate number of segments
        n_segments = max(2, n_segments)  # At least 2 segments
        
        boundary_frames = librosa.segment.agglomerative(S, n_segments)
        boundary_times = librosa.frames_to_time(boundary_frames, sr=sr)
        
        # Add start and end times
        if boundary_times[0] > 0:
            boundary_times = np.insert(boundary_times, 0, 0.0)
        
        duration = librosa.get_duration(y=y, sr=sr)
        if boundary_times[-1] < duration:
            boundary_times = np.append(boundary_times, duration)
        
        # Create sections with labels
        sections = []
        section_labels = ['Intro', 'Verse', 'Chorus', 'Bridge', 'Outro']
        
        # Simple heuristic for labeling sections
        for i in range(len(boundary_times) - 1):
            start_time = boundary_times[i]
            end_time = boundary_times[i+1]
            duration_sec = end_time - start_time
            
            # Very simple labeling heuristic
            if i == 0:
                label = 'Intro'
            elif i == len(boundary_times) - 2:
                label = 'Outro'
            elif i % 2 == 0:
                label = 'Verse'
            else:
                label = 'Chorus'
            
            # For longer sections in the middle, consider them bridges
            if i > 0 and i < len(boundary_times) - 2 and duration_sec > 30:
                label = 'Bridge'
            
            sections.append({
                'section': f"{label} {i//2 + 1}" if label in ['Verse', 'Chorus'] else label,
                'start': float(start_time),
                'end': float(end_time),
                'duration': float(duration_sec)
            })
        
        return {
            'sections': sections,
            'boundary_times': boundary_times.tolist(),
            'num_segments': len(sections)
        }
    
    def _generate_visualizations(self, y: np.ndarray, sr: int, audio_file: str) -> Dict[str, str]:
        """Generate visualizations for audio features."""
        logger.info("Generating visualizations")
        
        visualizations = {}
        base_filename = os.path.splitext(os.path.basename(audio_file))[0]
        
        # Waveform
        plt.figure(figsize=(10, 4))
        librosa.display.waveshow(y, sr=sr)
        plt.title('Waveform')
        plt.tight_layout()
        waveform_path = os.path.join(self.output_dir, f"{base_filename}_waveform.png")
        plt.savefig(waveform_path)
        plt.close()
        visualizations['waveform'] = waveform_path
        
        # Spectrogram
        plt.figure(figsize=(10, 4))
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Spectrogram')
        plt.tight_layout()
        spectrogram_path = os.path.join(self.output_dir, f"{base_filename}_spectrogram.png")
        plt.savefig(spectrogram_path)
        plt.close()
        visualizations['spectrogram'] = spectrogram_path
        
        # Chromagram
        plt.figure(figsize=(10, 4))
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        librosa.display.specshow(chroma, sr=sr, x_axis='time', y_axis='chroma')
        plt.colorbar()
        plt.title('Chromagram')
        plt.tight_layout()
        chroma_path = os.path.join(self.output_dir, f"{base_filename}_chroma.png")
        plt.savefig(chroma_path)
        plt.close()
        visualizations['chromagram'] = chroma_path
        
        # Mel spectrogram
        plt.figure(figsize=(10, 4))
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel Spectrogram')
        plt.tight_layout()
        mel_path = os.path.join(self.output_dir, f"{base_filename}_mel.png")
        plt.savefig(mel_path)
        plt.close()
        visualizations['mel_spectrogram'] = mel_path
        
        # Tempogram
        plt.figure(figsize=(10, 4))
        tempogram = librosa.feature.tempogram(y=y, sr=sr)
        librosa.display.specshow(tempogram, sr=sr, x_axis='time', y_axis='tempo')
        plt.colorbar()
        plt.title('Tempogram')
        plt.tight_layout()
        tempo_path = os.path.join(self.output_dir, f"{base_filename}_tempogram.png")
        plt.savefig(tempo_path)
        plt.close()
        visualizations['tempogram'] = tempo_path
        
        return visualizations
    
    def _make_json_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects to serializable types."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return obj
