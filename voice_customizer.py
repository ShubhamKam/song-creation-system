"""
Voice Customization Module for Streamlit Song Creation System

This module handles the customization of voice characteristics for generated songs.
"""

import os
import numpy as np
import tempfile
import logging
from typing import Dict, Any, Optional, List, Tuple
import json
import time
from pydub import AudioSegment
import librosa
import soundfile as sf

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VoiceCustomizer:
    """Class for customizing voice characteristics in generated songs."""
    
    def __init__(self, output_dir: str = None):
        """
        Initialize the voice customizer.
        
        Args:
            output_dir: Directory to save customized audio files
        """
        self.output_dir = output_dir or tempfile.gettempdir()
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"Voice customizer initialized with output directory: {self.output_dir}")
    
    def customize_voice(self, voice_sample_path: str, song_path: str, 
                       params: Dict[str, Any] = None,
                       progress_callback=None) -> Dict[str, Any]:
        """
        Apply voice customization to a generated song.
        
        Args:
            voice_sample_path: Path to the voice sample file
            song_path: Path to the generated song file
            params: Dictionary of voice customization parameters
            progress_callback: Optional callback function for progress updates
            
        Returns:
            Dictionary containing customization results and path to customized song
        """
        logger.info(f"Customizing voice for song: {song_path}")
        
        # Set default parameters if not provided
        if params is None:
            params = {}
        
        default_params = {
            'voice_pitch': 0,  # Semitones, -12 to 12
            'voice_timbre': 0.7,  # 0.0 to 1.0
            'voice_vibrato': 0.5,  # 0.0 to 1.0
            'voice_clarity': 0.8,  # 0.0 to 1.0
        }
        
        # Update defaults with provided parameters
        for key, value in params.items():
            if key in default_params:
                default_params[key] = value
        
        params = default_params
        
        try:
            # In a real implementation, this would use sophisticated voice synthesis
            # For this example, we'll simulate the process
            
            if progress_callback:
                progress_callback(0.1)  # 10% progress after initialization
            
            # Load voice sample
            logger.info(f"Loading voice sample: {voice_sample_path}")
            voice_y, voice_sr = librosa.load(voice_sample_path, sr=None)
            
            if progress_callback:
                progress_callback(0.3)  # 30% progress after loading voice
            
            # Load song
            logger.info(f"Loading song: {song_path}")
            song_y, song_sr = librosa.load(song_path, sr=None)
            
            if progress_callback:
                progress_callback(0.5)  # 50% progress after loading song
            
            # Extract voice characteristics
            logger.info("Extracting voice characteristics")
            voice_features = self._extract_voice_features(voice_y, voice_sr)
            
            if progress_callback:
                progress_callback(0.7)  # 70% progress after feature extraction
            
            # Apply voice customization
            logger.info("Applying voice customization")
            customized_y = self._apply_voice_customization(song_y, song_sr, voice_features, params)
            
            if progress_callback:
                progress_callback(0.9)  # 90% progress after customization
            
            # Save customized song
            output_file = os.path.join(
                self.output_dir, 
                f"voice_customized_{os.path.basename(song_path)}"
            )
            
            sf.write(output_file, customized_y, song_sr)
            
            # Prepare result info
            result = {
                'original_voice_sample': voice_sample_path,
                'original_song': song_path,
                'customized_song': output_file,
                'voice_features': voice_features,
                'customization_params': params,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Save result info to JSON
            info_file = os.path.join(self.output_dir, os.path.basename(output_file) + '_info.json')
            with open(info_file, 'w') as f:
                json.dump(result, f, indent=2)
            
            if progress_callback:
                progress_callback(1.0)  # 100% progress
            
            logger.info(f"Voice customization completed: {output_file}")
            return result
            
        except Exception as e:
            logger.error(f"Error customizing voice: {str(e)}")
            raise
    
    def _extract_voice_features(self, voice_y: np.ndarray, voice_sr: int) -> Dict[str, Any]:
        """Extract key features from voice sample."""
        logger.info("Extracting voice features")
        
        # In a real implementation, this would use sophisticated voice analysis
        # For this example, we'll extract some basic features
        
        # Extract pitch
        pitches, magnitudes = librosa.piptrack(y=voice_y, sr=voice_sr)
        pitch_indices = np.argmax(magnitudes, axis=0)
        pitches = pitches[pitch_indices, np.arange(magnitudes.shape[1])]
        pitches = pitches[pitches > 0]
        mean_pitch = np.mean(pitches) if len(pitches) > 0 else 0
        
        # Extract timbre (using MFCCs)
        mfccs = librosa.feature.mfcc(y=voice_y, sr=voice_sr, n_mfcc=13)
        mean_mfccs = np.mean(mfccs, axis=1)
        
        # Extract vibrato (frequency modulation)
        # Simplified approach: measure pitch variation
        pitch_std = np.std(pitches) if len(pitches) > 0 else 0
        vibrato = min(1.0, pitch_std / 10.0)  # Normalize to 0-1
        
        # Extract clarity (using spectral contrast)
        contrast = librosa.feature.spectral_contrast(y=voice_y, sr=voice_sr)
        mean_contrast = np.mean(contrast)
        clarity = min(1.0, mean_contrast / 10.0)  # Normalize to 0-1
        
        return {
            'mean_pitch': float(mean_pitch),
            'mfccs': mean_mfccs.tolist(),
            'vibrato': float(vibrato),
            'clarity': float(clarity)
        }
    
    def _apply_voice_customization(self, song_y: np.ndarray, song_sr: int, 
                                 voice_features: Dict[str, Any],
                                 params: Dict[str, Any]) -> np.ndarray:
        """Apply voice customization to song."""
        logger.info("Applying voice customization")
        
        # In a real implementation, this would use sophisticated voice synthesis
        # For this example, we'll simulate the process with basic audio processing
        
        # Make a copy of the song
        customized_y = np.copy(song_y)
        
        # Apply pitch shift
        if params['voice_pitch'] != 0:
            customized_y = librosa.effects.pitch_shift(
                customized_y, 
                sr=song_sr, 
                n_steps=params['voice_pitch']
            )
        
        # Apply timbre modification (simplified)
        # In a real implementation, this would use more sophisticated techniques
        if params['voice_timbre'] > 0.5:
            # Simulate timbre change by adjusting frequency content
            D = librosa.stft(customized_y)
            
            # Apply a simple filter based on voice timbre
            filter_strength = (params['voice_timbre'] - 0.5) * 2  # 0 to 1
            
            # Create a simple filter based on voice MFCCs (simplified)
            filter_shape = np.ones(D.shape[0])
            for i in range(min(len(filter_shape), len(voice_features['mfccs']))):
                filter_shape[i] = 1.0 + filter_strength * (voice_features['mfccs'][i] / 10.0)
            
            # Apply filter
            for i in range(min(len(filter_shape), D.shape[0])):
                D[i, :] = D[i, :] * filter_shape[i]
            
            # Convert back to time domain
            customized_y = librosa.istft(D)
        
        # Apply vibrato (simplified)
        if params['voice_vibrato'] > 0.2:
            # Simulate vibrato by applying slight pitch modulation
            vibrato_rate = 5.0  # Hz
            vibrato_depth = params['voice_vibrato'] * 0.3  # Depth in semitones
            
            # Create modulation signal
            t = np.arange(len(customized_y)) / song_sr
            mod = np.sin(2 * np.pi * vibrato_rate * t) * vibrato_depth
            
            # Apply modulation (simplified approach)
            # In a real implementation, this would use more sophisticated techniques
            customized_y = librosa.effects.pitch_shift(
                customized_y, 
                sr=song_sr, 
                n_steps=mod[0]  # Just use the first value as an example
            )
        
        # Apply clarity enhancement (simplified)
        if params['voice_clarity'] > 0.5:
            # Simulate clarity enhancement with simple EQ
            D = librosa.stft(customized_y)
            
            # Enhance high frequencies for clarity
            clarity_strength = (params['voice_clarity'] - 0.5) * 2  # 0 to 1
            
            # Simple high-shelf filter
            freq_bins = D.shape[0]
            high_shelf = np.ones(freq_bins)
            
            # Boost high frequencies
            high_shelf[freq_bins//2:] = 1.0 + clarity_strength
            
            # Apply filter
            for i in range(D.shape[0]):
                D[i, :] = D[i, :] * high_shelf[i]
            
            # Convert back to time domain
            customized_y = librosa.istft(D)
        
        return customized_y
