"""
Song Generation Module for Streamlit Song Creation System

This module handles the generation of new songs based on analyzed features.
"""

import os
import numpy as np
import tempfile
import logging
from typing import Dict, Any, Optional, List, Tuple
import json
import midiutil
from pydub import AudioSegment
import random
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SongGenerator:
    """Class for generating songs based on analyzed features."""
    
    def __init__(self, output_dir: str = None):
        """
        Initialize the song generator.
        
        Args:
            output_dir: Directory to save generated songs
        """
        self.output_dir = output_dir or tempfile.gettempdir()
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"Song generator initialized with output directory: {self.output_dir}")
    
    def generate_song(self, analysis_results: Dict[str, Any], 
                     params: Dict[str, Any] = None,
                     progress_callback=None) -> Dict[str, Any]:
        """
        Generate a song based on analysis results.
        
        Args:
            analysis_results: Dictionary containing audio analysis results
            params: Dictionary of generation parameters
            progress_callback: Optional callback function for progress updates
            
        Returns:
            Dictionary containing generation results and paths to generated files
        """
        logger.info("Generating song based on analysis results")
        
        # Set default parameters if not provided
        if params is None:
            params = {}
        
        default_params = {
            'model': 'neural_network',
            'creativity': 0.7,
            'similarity': 0.5,
            'duration': 180,
            'tempo_variation': 0,
            'key_shift': 0,
            'rhythm_complexity': 0.5,
            'harmonic_complexity': 0.5,
            'structure_variation': 0.3,
            'lyric_influence': 0.6
        }
        
        # Update defaults with provided parameters
        for key, value in params.items():
            if key in default_params:
                default_params[key] = value
        
        params = default_params
        
        # Extract key features from analysis results
        try:
            # Get rhythm features
            tempo = analysis_results.get('rhythm', {}).get('tempo', 120)
            
            # Apply tempo variation
            tempo = tempo + params['tempo_variation']
            
            # Get harmony features
            key = analysis_results.get('harmony', {}).get('key', 'C')
            mode = analysis_results.get('harmony', {}).get('mode', 'Major')
            
            # Apply key shift
            key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            key_index = key_names.index(key)
            shifted_key_index = (key_index + params['key_shift']) % 12
            key = key_names[shifted_key_index]
            
            # Get structure
            sections = analysis_results.get('structure', {}).get('sections', [])
            
            # If no sections, create a default structure
            if not sections:
                sections = [
                    {'section': 'Intro', 'duration': 15},
                    {'section': 'Verse 1', 'duration': 30},
                    {'section': 'Chorus', 'duration': 30},
                    {'section': 'Verse 2', 'duration': 30},
                    {'section': 'Chorus', 'duration': 30},
                    {'section': 'Bridge', 'duration': 30},
                    {'section': 'Chorus', 'duration': 30},
                    {'section': 'Outro', 'duration': 15}
                ]
            
            # Apply structure variation
            if params['structure_variation'] > 0.5:
                # Shuffle some sections
                verse_indices = [i for i, s in enumerate(sections) if 'Verse' in s['section']]
                if len(verse_indices) >= 2:
                    i, j = random.sample(verse_indices, 2)
                    sections[i], sections[j] = sections[j], sections[i]
                
                # Maybe add or remove a section
                if random.random() < params['structure_variation'] - 0.5:
                    if len(sections) > 3 and random.random() < 0.5:
                        # Remove a non-essential section
                        non_essential = [i for i, s in enumerate(sections) 
                                        if s['section'] not in ['Intro', 'Outro', 'Chorus']]
                        if non_essential:
                            sections.pop(random.choice(non_essential))
                    else:
                        # Add a section
                        new_section = {'section': 'Bridge 2', 'duration': 20}
                        insert_pos = random.randint(2, len(sections) - 1)
                        sections.insert(insert_pos, new_section)
            
            if progress_callback:
                progress_callback(0.1)  # 10% progress after preparation
            
            # Generate MIDI based on model type
            if params['model'] == 'neural_network':
                midi_file = self._generate_neural_network(
                    tempo, key, mode, sections, params, progress_callback
                )
            elif params['model'] == 'markov_chain':
                midi_file = self._generate_markov_chain(
                    tempo, key, mode, sections, params, progress_callback
                )
            elif params['model'] == 'transformer':
                midi_file = self._generate_transformer(
                    tempo, key, mode, sections, params, progress_callback
                )
            else:  # Default to hybrid
                midi_file = self._generate_hybrid(
                    tempo, key, mode, sections, params, progress_callback
                )
            
            if progress_callback:
                progress_callback(0.7)  # 70% progress after MIDI generation
            
            # Convert MIDI to audio
            audio_file = self._midi_to_audio(midi_file)
            
            if progress_callback:
                progress_callback(0.9)  # 90% progress after audio conversion
            
            # Prepare generation info
            generation_info = {
                'model': params['model'],
                'tempo': tempo,
                'key': key,
                'mode': mode,
                'duration': params['duration'],
                'creativity': params['creativity'],
                'similarity': params['similarity'],
                'midi_file': midi_file,
                'audio_file': audio_file,
                'generation_params': params,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Save generation info to JSON
            info_file = os.path.join(self.output_dir, os.path.basename(audio_file) + '_info.json')
            with open(info_file, 'w') as f:
                json.dump(generation_info, f, indent=2)
            
            if progress_callback:
                progress_callback(1.0)  # 100% progress
            
            logger.info(f"Song generation completed: {audio_file}")
            return generation_info
            
        except Exception as e:
            logger.error(f"Error generating song: {str(e)}")
            raise
    
    def _generate_neural_network(self, tempo, key, mode, sections, params, progress_callback=None) -> str:
        """Generate song using neural network approach (simplified simulation)."""
        logger.info("Generating song using neural network model")
        
        # Create a MIDIFile object with 1 track
        midi = midiutil.MIDIFile(1)
        
        # Track name and tempo
        track = 0
        time = 0
        midi.addTrackName(track, time, "Neural Network Generated Track")
        midi.addTempo(track, time, tempo)
        
        # Define key mapping
        key_map = {
            'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5,
            'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11
        }
        
        # Define scale based on key and mode
        root = key_map[key]
        if mode == 'Major':
            scale_intervals = [0, 2, 4, 5, 7, 9, 11]  # Major scale intervals
        else:
            scale_intervals = [0, 2, 3, 5, 7, 8, 10]  # Minor scale intervals
        
        scale = [(root + interval) % 12 + 60 for interval in scale_intervals]
        
        # Define chord progressions for different sections
        progressions = {
            'Intro': [0, 3, 4, 3],  # I-IV-V-IV
            'Verse': [0, 3, 4, 3, 0, 3, 4, 3],  # I-IV-V-IV-I-IV-V-IV
            'Chorus': [3, 0, 5, 4, 3, 0, 5, 4],  # IV-I-vi-V-IV-I-vi-V
            'Bridge': [5, 4, 3, 0, 5, 4, 3, 0],  # vi-V-IV-I-vi-V-IV-I
            'Outro': [0, 5, 3, 4, 0]  # I-vi-IV-V-I
        }
        
        # Current time position in quarter notes
        current_time = 0
        
        # Process each section
        for section_idx, section in enumerate(sections):
            section_name = section['section'].split()[0] if ' ' in section['section'] else section['section']
            section_duration = section.get('duration', 30)  # Default 30 seconds if not specified
            
            # Calculate number of measures based on duration and tempo
            # Assuming 4/4 time signature
            beats_per_second = tempo / 60
            total_beats = section_duration * beats_per_second
            measures = int(total_beats / 4)  # 4 beats per measure in 4/4
            
            # Ensure at least one measure
            measures = max(1, measures)
            
            # Get progression for this section type
            progression = progressions.get(section_name, progressions['Verse'])
            
            # Repeat progression as needed to fill measures
            full_progression = []
            while len(full_progression) < measures:
                full_progression.extend(progression)
            full_progression = full_progression[:measures]
            
            # For each measure, add a chord
            for measure_idx, chord_idx in enumerate(full_progression):
                # Get chord notes (triad)
                chord_root = scale[chord_idx]
                chord_third = scale[(chord_idx + 2) % 7]
                chord_fifth = scale[(chord_idx + 4) % 7]
                
                # Add bass note
                midi.addNote(track, 0, chord_root - 12, current_time, 4, 100)
                
                # Add chord notes with some rhythm variation
                if params['rhythm_complexity'] < 0.3:
                    # Simple rhythm - whole notes
                    midi.addNote(track, 0, chord_root, current_time, 4, 80)
                    midi.addNote(track, 0, chord_third, current_time, 4, 80)
                    midi.addNote(track, 0, chord_fifth, current_time, 4, 80)
                elif params['rhythm_complexity'] < 0.7:
                    # Medium rhythm - half notes
                    midi.addNote(track, 0, chord_root, current_time, 2, 80)
                    midi.addNote(track, 0, chord_third, current_time, 2, 80)
                    midi.addNote(track, 0, chord_fifth, current_time, 2, 80)
                    midi.addNote(track, 0, chord_root, current_time + 2, 2, 80)
                    midi.addNote(track, 0, chord_third, current_time + 2, 2, 80)
                    midi.addNote(track, 0, chord_fifth, current_time + 2, 2, 80)
                else:
                    # Complex rhythm - quarter notes with some variation
                    for beat in range(4):
                        if random.random() < 0.8:  # 80% chance to play a note
                            midi.addNote(track, 0, chord_root, current_time + beat, 1, 80)
                        if random.random() < 0.7:  # 70% chance to play a note
                            midi.addNote(track, 0, chord_third, current_time + beat, 1, 80)
                        if random.random() < 0.6:  # 60% chance to play a note
                            midi.addNote(track, 0, chord_fifth, current_time + beat, 1, 80)
                
                # Add melody notes
                melody_notes = []
                for beat in range(4):
                    # Determine if we should add a melody note
                    if random.random() < 0.7:  # 70% chance for a melody note
                        # Choose a note from the scale
                        note_idx = random.randint(0, 6)
                        note = scale[note_idx] + 12  # Octave higher
                        
                        # Determine duration (1, 0.5, or 0.25 beats)
                        durations = [1, 0.5, 0.25]
                        weights = [0.6, 0.3, 0.1]  # Probability weights
                        duration = random.choices(durations, weights)[0]
                        
                        # Add the note if it fits within the measure
                        if current_time + beat + duration <= current_time + 4:
                            midi.addNote(track, 0, note, current_time + beat, duration, 100)
                            melody_notes.append((note, current_time + beat, duration))
                
                # Move to next measure
                current_time += 4
            
            # Update progress if callback provided
            if progress_callback and len(sections) > 0:
                progress = 0.1 + 0.6 * (section_idx + 1) / len(sections)
                progress_callback(min(progress, 0.7))
        
        # Write the MIDI file
        midi_file = os.path.join(self.output_dir, f"neural_network_generated_{int(time.time())}.mid")
        with open(midi_file, 'wb') as output_file:
            midi.writeFile(output_file)
        
        return midi_file
    
    def _generate_markov_chain(self, tempo, key, mode, sections, params, progress_callback=None) -> str:
        """Generate song using Markov chain approach (simplified simulation)."""
        logger.info("Generating song using Markov chain model")
        
        # Similar implementation to neural network but with different patterns
        # For simplicity, we'll reuse the neural network implementation with slight modifications
        
        # Create a MIDIFile object with 1 track
        midi = midiutil.MIDIFile(1)
        
        # Track name and tempo
        track = 0
        time = 0
        midi.addTrackName(track, time, "Markov Chain Generated Track")
        midi.addTempo(track, time, tempo)
        
        # Define key mapping
        key_map = {
            'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5,
            'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11
        }
        
        # Define scale based on key and mode
        root = key_map[key]
        if mode == 'Major':
            scale_intervals = [0, 2, 4, 5, 7, 9, 11]  # Major scale intervals
        else:
            scale_intervals = [0, 2, 3, 5, 7, 8, 10]  # Minor scale intervals
        
        scale = [(root + interval) % 12 + 60 for interval in scale_intervals]
        
        # Define Markov chain transition probabilities for chord progressions
        # Format: {current_chord_index: {next_chord_index: probability}}
        transition_probs = {
            0: {0: 0.1, 1: 0.1, 2: 0.1, 3: 0.3, 4: 0.3, 5: 0.1},  # From I
            1: {0: 0.3, 1: 0.1, 2: 0.2, 3: 0.1, 4: 0.2, 5: 0.1},  # From ii
            2: {0: 0.2, 1: 0.1, 2: 0.1, 3: 0.2, 4: 0.3, 5: 0.1},  # From iii
            3: {0: 0.3, 1: 0.1, 2: 0.1, 3: 0.1, 4: 0.3, 5: 0.1},  # From IV
            4: {0: 0.5, 1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1, 5: 0.1},  # From V
            5: {0: 0.1, 1: 0.1, 2: 0.1, 3: 0.2, 4: 0.4, 5: 0.1}   # From vi
        }
        
        # Current time position in quarter notes
        current_time = 0
        
        # Process each section
        for section_idx, section in enumerate(sections):
            section_name = section['section'].split()[0] if ' ' in section['section'] else section['section']
            section_duration = section.get('duration', 30)  # Default 30 seconds if not specified
            
            # Calculate number of measures based on duration and tempo
            # Assuming 4/4 time signature
            beats_per_second = tempo / 60
            total_beats = section_duration * beats_per_second
            measures = int(total_beats / 4)  # 4 beats per measure in 4/4
            
            # Ensure at least one measure
            measures = max(1, measures)
            
            # Generate chord progression using Markov chain
            progression = []
            current_chord = 0  # Start with I chord
            
            for _ in range(measures):
                progression.append(current_chord)
                
                # Get next chord based on transition probabilities
                next_chord_probs = transition_probs[current_chord]
                next_chord_options = list(next_chord_probs.keys())
                next_chord_weights = list(next_chord_probs.values())
                
                current_chord = random.choices(next_chord_options, next_chord_weights)[0]
            
            # For each measure, add a chord
            for measure_idx, chord_idx in enumerate(progression):
                # Get chord notes (triad)
                chord_root = scale[chord_idx]
                chord_third = scale[(chord_idx + 2) % 7]
                chord_fifth = scale[(chord_idx + 4) % 7]
                
                # Add bass note
                midi.addNote(track, 0, chord_root - 12, current_time, 4, 100)
                
                # Add chord notes with some rhythm variation
                if params['rhythm_complexity'] < 0.3:
                    # Simple rhythm - whole notes
                    midi.addNote(track, 0, chord_root, current_time, 4, 80)
                    midi.addNote(track, 0, chord_third, current_time, 4, 80)
                    midi.addNote(track, 0, chord_fifth, current_time, 4, 80)
                elif params['rhythm_complexity'] < 0.7:
                    # Medium rhythm - half notes
                    midi.addNote(track, 0, chord_root, current_time, 2, 80)
                    midi.addNote(track, 0, chord_third, current_time, 2, 80)
                    midi.addNote(track, 0, chord_fifth, current_time, 2, 80)
                    midi.addNote(track, 0, chord_root, current_time + 2, 2, 80)
                    midi.addNote(track, 0, chord_third, current_time + 2, 2, 80)
                    midi.addNote(track, 0, chord_fifth, current_time + 2, 2, 80)
                else:
                    # Complex rhythm - quarter notes with some variation
                    for beat in range(4):
                        if random.random() < 0.8:  # 80% chance to play a note
                            midi.addNote(track, 0, chord_root, current_time + beat, 1, 80)
                        if random.random() < 0.7:  # 70% chance to play a note
                            midi.addNote(track, 0, chord_third, current_time + beat, 1, 80)
                        if random.random() < 0.6:  # 60% chance to play a note
                            midi.addNote(track, 0, chord_fifth, current_time + beat, 1, 80)
                
                # Add melody notes using Markov approach
                if measure_idx > 0:
                    # Get the last melody note from previous measure
                    last_note = random.choice(scale)
                    
                    # Define transition probabilities for melody
                    # Higher probability for stepwise motion
                    for beat in range(4):
                        if random.random() < 0.8:  # 80% chance for a melody note
                            # Determine the next note based on the last note
                            note_idx = scale.index(last_note % 12 + 60)
                            
                            # Possible transitions: up/down by step, up/down by third, stay same
                            transitions = [-2, -1, 0, 1, 2]
                            weights = [0.1, 0.3, 0.2, 0.3, 0.1]  # Favor stepwise motion
                            
                            # Apply transition
                            new_idx = (note_idx + random.choices(transitions, weights)[0]) % 7
                            note = scale[new_idx] + 12  # Octave higher
                            
                            # Update last note
                            last_note = note
                            
                            # Determine duration (1, 0.5, or 0.25 beats)
                            durations = [1, 0.5, 0.25]
                            weights = [0.5, 0.3, 0.2]  # Probability weights
                            duration = random.choices(durations, weights)[0]
                            
                            # Add the note if it fits within the measure
                            if current_time + beat + duration <= current_time + 4:
                                midi.addNote(track, 0, note, current_time + beat, duration, 100)
                
                # Move to next measure
                current_time += 4
            
            # Update progress if callback provided
            if progress_callback and len(sections) > 0:
                progress = 0.1 + 0.6 * (section_idx + 1) / len(sections)
                progress_callback(min(progress, 0.7))
        
        # Write the MIDI file
        midi_file = os.path.join(self.output_dir, f"markov_chain_generated_{int(time.time())}.mid")
        with open(midi_file, 'wb') as output_file:
            midi.writeFile(output_file)
        
        return midi_file
    
    def _generate_transformer(self, tempo, key, mode, sections, params, progress_callback=None) -> str:
        """Generate song using transformer approach (simplified simulation)."""
        logger.info("Generating song using transformer model")
        
        # For simplicity, we'll implement a variation of the neural network approach
        # In a real implementation, this would use a transformer model
        
        # Create a MIDIFile object with 1 track
        midi = midiutil.MIDIFile(1)
        
        # Track name and tempo
        track = 0
        time = 0
        midi.addTrackName(track, time, "Transformer Generated Track")
        midi.addTempo(track, time, tempo)
        
        # Define key mapping
        key_map = {
            'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5,
            'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11
        }
        
        # Define scale based on key and mode
        root = key_map[key]
        if mode == 'Major':
            scale_intervals = [0, 2, 4, 5, 7, 9, 11]  # Major scale intervals
        else:
            scale_intervals = [0, 2, 3, 5, 7, 8, 10]  # Minor scale intervals
        
        scale = [(root + interval) % 12 + 60 for interval in scale_intervals]
        
        # Define more complex chord progressions for different sections
        progressions = {
            'Intro': [0, 5, 3, 4],  # I-vi-IV-V
            'Verse': [0, 5, 3, 4, 0, 5, 1, 4],  # I-vi-IV-V-I-vi-ii-V
            'Chorus': [3, 0, 5, 4, 3, 0, 5, 4],  # IV-I-vi-V-IV-I-vi-V
            'Bridge': [5, 1, 2, 3, 4, 0],  # vi-ii-iii-IV-V-I
            'Outro': [0, 3, 4, 0]  # I-IV-V-I
        }
        
        # Current time position in quarter notes
        current_time = 0
        
        # Process each section
        for section_idx, section in enumerate(sections):
            section_name = section['section'].split()[0] if ' ' in section['section'] else section['section']
            section_duration = section.get('duration', 30)  # Default 30 seconds if not specified
            
            # Calculate number of measures based on duration and tempo
            # Assuming 4/4 time signature
            beats_per_second = tempo / 60
            total_beats = section_duration * beats_per_second
            measures = int(total_beats / 4)  # 4 beats per measure in 4/4
            
            # Ensure at least one measure
            measures = max(1, measures)
            
            # Get progression for this section type
            progression = progressions.get(section_name, progressions['Verse'])
            
            # Repeat progression as needed to fill measures
            full_progression = []
            while len(full_progression) < measures:
                full_progression.extend(progression)
            full_progression = full_progression[:measures]
            
            # For each measure, add a chord
            for measure_idx, chord_idx in enumerate(full_progression):
                # Get chord notes (triad)
                chord_root = scale[chord_idx]
                chord_third = scale[(chord_idx + 2) % 7]
                chord_fifth = scale[(chord_idx + 4) % 7]
                
                # For transformer model, add more complex chords
                # Add 7th or 9th based on harmonic complexity
                if params['harmonic_complexity'] > 0.6:
                    chord_seventh = scale[(chord_idx + 6) % 7]
                    if params['harmonic_complexity'] > 0.8:
                        chord_ninth = scale[(chord_idx + 8) % 7]
                
                # Add bass note with rhythm
                if random.random() < 0.7:
                    midi.addNote(track, 0, chord_root - 12, current_time, 2, 100)
                    midi.addNote(track, 0, chord_root - 12, current_time + 2, 2, 90)
                else:
                    midi.addNote(track, 0, chord_root - 12, current_time, 1, 100)
                    midi.addNote(track, 0, chord_fifth - 12, current_time + 1, 1, 90)
                    midi.addNote(track, 0, chord_root - 12, current_time + 2, 1, 95)
                    midi.addNote(track, 0, chord_fifth - 12, current_time + 3, 1, 85)
                
                # Add chord notes with complex rhythm patterns
                # Transformer models would generate more sophisticated patterns
                if params['rhythm_complexity'] < 0.4:
                    # Simple rhythm but with some variation
                    for i in range(4):
                        if i % 2 == 0:
                            midi.addNote(track, 0, chord_root, current_time + i, 1, 80)
                            midi.addNote(track, 0, chord_third, current_time + i, 1, 80)
                            midi.addNote(track, 0, chord_fifth, current_time + i, 1, 80)
                        else:
                            midi.addNote(track, 0, chord_root, current_time + i, 1, 70)
                            midi.addNote(track, 0, chord_fifth, current_time + i, 1, 70)
                elif params['rhythm_complexity'] < 0.7:
                    # Medium complexity rhythm
                    patterns = [
                        [0.5, 0.5, 1, 1, 1],
                        [1, 0.5, 0.5, 1, 1],
                        [1, 1, 0.5, 0.5, 1]
                    ]
                    pattern = random.choice(patterns)
                    
                    pos = 0
                    for duration in pattern:
                        if pos < 4:  # Stay within the measure
                            midi.addNote(track, 0, chord_root, current_time + pos, duration, 80)
                            midi.addNote(track, 0, chord_third, current_time + pos, duration, 80)
                            midi.addNote(track, 0, chord_fifth, current_time + pos, duration, 80)
                            pos += duration
                else:
                    # Complex rhythm
                    for beat in range(16):  # 16 sixteenth notes in a measure
                        # More complex rhythmic pattern
                        if beat % 4 == 0:  # Strong beats
                            midi.addNote(track, 0, chord_root, current_time + beat/4, 0.25, 85)
                            midi.addNote(track, 0, chord_third, current_time + beat/4, 0.25, 85)
                            midi.addNote(track, 0, chord_fifth, current_time + beat/4, 0.25, 85)
                        elif beat % 2 == 0:  # Medium beats
                            if random.random() < 0.8:
                                midi.addNote(track, 0, chord_root, current_time + beat/4, 0.25, 75)
                                midi.addNote(track, 0, chord_fifth, current_time + beat/4, 0.25, 75)
                        else:  # Weak beats
                            if random.random() < 0.4:
                                midi.addNote(track, 0, chord_third, current_time + beat/4, 0.25, 65)
                
                # Add melody with more sophisticated patterns
                # Transformer would generate coherent melodies with motifs
                melody_sequence = []
                
                # Create a motif for this section if it's the first measure
                if measure_idx % len(progression) == 0:
                    motif = []
                    # Generate a short motif (2-4 notes)
                    motif_length = random.randint(2, 4)
                    for _ in range(motif_length):
                        note_idx = random.randint(0, 6)
                        note = scale[note_idx] + 12  # Octave higher
                        duration = random.choice([0.25, 0.5, 1])
                        motif.append((note, duration))
                
                # Use the motif with variations
                if measure_idx % 2 == 0:  # Original motif
                    pos = 0
                    for note, duration in motif:
                        if pos < 4:  # Stay within measure
                            midi.addNote(track, 0, note, current_time + pos, duration, 100)
                            pos += duration
                else:  # Variation of motif
                    pos = 0
                    for note, duration in motif:
                        if pos < 4:  # Stay within measure
                            # Modify note slightly
                            variation = random.choice([-2, -1, 0, 1, 2])
                            note_idx = scale.index(note % 12 + 60)
                            new_idx = (note_idx + variation) % 7
                            new_note = scale[new_idx] + 12
                            
                            midi.addNote(track, 0, new_note, current_time + pos, duration, 100)
                            pos += duration
                
                # Move to next measure
                current_time += 4
            
            # Update progress if callback provided
            if progress_callback and len(sections) > 0:
                progress = 0.1 + 0.6 * (section_idx + 1) / len(sections)
                progress_callback(min(progress, 0.7))
        
        # Write the MIDI file
        midi_file = os.path.join(self.output_dir, f"transformer_generated_{int(time.time())}.mid")
        with open(midi_file, 'wb') as output_file:
            midi.writeFile(output_file)
        
        return midi_file
    
    def _generate_hybrid(self, tempo, key, mode, sections, params, progress_callback=None) -> str:
        """Generate song using hybrid approach combining multiple models."""
        logger.info("Generating song using hybrid model")
        
        # For simplicity, we'll implement a combination of the previous approaches
        # In a real implementation, this would use multiple models
        
        # Randomly choose which model to use for each section
        models = ['neural_network', 'markov_chain', 'transformer']
        
        # Create a MIDIFile object with 1 track
        midi = midiutil.MIDIFile(1)
        
        # Track name and tempo
        track = 0
        time = 0
        midi.addTrackName(track, time, "Hybrid Generated Track")
        midi.addTempo(track, time, tempo)
        
        # Define key mapping
        key_map = {
            'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5,
            'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11
        }
        
        # Define scale based on key and mode
        root = key_map[key]
        if mode == 'Major':
            scale_intervals = [0, 2, 4, 5, 7, 9, 11]  # Major scale intervals
        else:
            scale_intervals = [0, 2, 3, 5, 7, 8, 10]  # Minor scale intervals
        
        scale = [(root + interval) % 12 + 60 for interval in scale_intervals]
        
        # Current time position in quarter notes
        current_time = 0
        
        # Process each section
        for section_idx, section in enumerate(sections):
            # Choose a model for this section
            model = random.choice(models)
            
            section_name = section['section'].split()[0] if ' ' in section['section'] else section['section']
            section_duration = section.get('duration', 30)  # Default 30 seconds if not specified
            
            # Calculate number of measures based on duration and tempo
            # Assuming 4/4 time signature
            beats_per_second = tempo / 60
            total_beats = section_duration * beats_per_second
            measures = int(total_beats / 4)  # 4 beats per measure in 4/4
            
            # Ensure at least one measure
            measures = max(1, measures)
            
            # Generate section based on chosen model
            if model == 'neural_network':
                # Neural network-style progression
                progression = [0, 3, 4, 3] * (measures // 4 + 1)  # I-IV-V-IV
            elif model == 'markov_chain':
                # Markov chain-style progression
                progression = []
                current_chord = 0  # Start with I chord
                
                # Simple transition matrix
                transitions = {
                    0: [0, 3, 4, 5],  # From I
                    3: [0, 3, 4, 5],  # From IV
                    4: [0, 3, 4, 5],  # From V
                    5: [0, 3, 4, 5]   # From vi
                }
                
                for _ in range(measures):
                    progression.append(current_chord)
                    current_chord = random.choice(transitions[current_chord])
            else:  # transformer
                # Transformer-style progression
                progressions = {
                    'Intro': [0, 5, 3, 4],  # I-vi-IV-V
                    'Verse': [0, 5, 3, 4, 0, 5, 1, 4],  # I-vi-IV-V-I-vi-ii-V
                    'Chorus': [3, 0, 5, 4, 3, 0, 5, 4],  # IV-I-vi-V-IV-I-vi-V
                    'Bridge': [5, 1, 2, 3, 4, 0],  # vi-ii-iii-IV-V-I
                    'Outro': [0, 3, 4, 0]  # I-IV-V-I
                }
                
                section_prog = progressions.get(section_name, progressions['Verse'])
                progression = section_prog * (measures // len(section_prog) + 1)
            
            # Trim to exact number of measures
            progression = progression[:measures]
            
            # For each measure, add a chord
            for measure_idx, chord_idx in enumerate(progression):
                # Get chord notes (triad)
                chord_root = scale[chord_idx]
                chord_third = scale[(chord_idx + 2) % 7]
                chord_fifth = scale[(chord_idx + 4) % 7]
                
                # Add bass note
                midi.addNote(track, 0, chord_root - 12, current_time, 4, 100)
                
                # Add chord notes with rhythm based on model
                if model == 'neural_network':
                    # Simple rhythm
                    midi.addNote(track, 0, chord_root, current_time, 4, 80)
                    midi.addNote(track, 0, chord_third, current_time, 4, 80)
                    midi.addNote(track, 0, chord_fifth, current_time, 4, 80)
                elif model == 'markov_chain':
                    # Medium rhythm
                    midi.addNote(track, 0, chord_root, current_time, 2, 80)
                    midi.addNote(track, 0, chord_third, current_time, 2, 80)
                    midi.addNote(track, 0, chord_fifth, current_time, 2, 80)
                    midi.addNote(track, 0, chord_root, current_time + 2, 2, 80)
                    midi.addNote(track, 0, chord_third, current_time + 2, 2, 80)
                    midi.addNote(track, 0, chord_fifth, current_time + 2, 2, 80)
                else:  # transformer
                    # Complex rhythm
                    for beat in range(4):
                        if beat % 2 == 0:  # Strong beats
                            midi.addNote(track, 0, chord_root, current_time + beat, 1, 80)
                            midi.addNote(track, 0, chord_third, current_time + beat, 1, 80)
                            midi.addNote(track, 0, chord_fifth, current_time + beat, 1, 80)
                        else:  # Weak beats
                            if random.random() < 0.7:
                                midi.addNote(track, 0, chord_root, current_time + beat, 1, 70)
                                midi.addNote(track, 0, chord_fifth, current_time + beat, 1, 70)
                
                # Add melody based on model
                if model == 'neural_network':
                    # Simple melody
                    for beat in range(4):
                        if random.random() < 0.6:
                            note_idx = random.randint(0, 6)
                            note = scale[note_idx] + 12
                            midi.addNote(track, 0, note, current_time + beat, 1, 100)
                elif model == 'markov_chain':
                    # Stepwise melody
                    current_note_idx = random.randint(0, 6)
                    for beat in range(4):
                        if random.random() < 0.7:
                            # Move up or down by step
                            step = random.choice([-1, 0, 1])
                            current_note_idx = (current_note_idx + step) % 7
                            note = scale[current_note_idx] + 12
                            midi.addNote(track, 0, note, current_time + beat, 1, 100)
                else:  # transformer
                    # More complex melody with motifs
                    if measure_idx % 2 == 0:
                        # Create a motif
                        motif = []
                        pos = 0
                        while pos < 4:
                            note_idx = random.randint(0, 6)
                            note = scale[note_idx] + 12
                            duration = random.choice([0.5, 1])
                            if pos + duration <= 4:
                                midi.addNote(track, 0, note, current_time + pos, duration, 100)
                                motif.append((note, current_time + pos, duration))
                                pos += duration
                    else:
                        # Repeat motif with variation
                        for note, time_pos, duration in motif:
                            # Transpose motif
                            note_idx = scale.index(note % 12 + 60)
                            new_idx = (note_idx + random.choice([-1, 0, 1])) % 7
                            new_note = scale[new_idx] + 12
                            
                            # Adjust timing
                            new_pos = time_pos - current_time + 4
                            if 0 <= new_pos < 4:
                                midi.addNote(track, 0, new_note, current_time + new_pos, duration, 100)
                
                # Move to next measure
                current_time += 4
            
            # Update progress if callback provided
            if progress_callback and len(sections) > 0:
                progress = 0.1 + 0.6 * (section_idx + 1) / len(sections)
                progress_callback(min(progress, 0.7))
        
        # Write the MIDI file
        midi_file = os.path.join(self.output_dir, f"hybrid_generated_{int(time.time())}.mid")
        with open(midi_file, 'wb') as output_file:
            midi.writeFile(output_file)
        
        return midi_file
    
    def _midi_to_audio(self, midi_file: str) -> str:
        """Convert MIDI file to audio (MP3)."""
        logger.info(f"Converting MIDI to audio: {midi_file}")
        
        # In a real implementation, this would use a synthesizer like FluidSynth
        # For this example, we'll just create a dummy audio file
        
        # Get the base filename without extension
        base_name = os.path.splitext(os.path.basename(midi_file))[0]
        audio_file = os.path.join(self.output_dir, f"{base_name}.mp3")
        
        # In a real implementation, we would convert MIDI to audio here
        # For now, we'll just create a dummy audio file
        
        # Create a simple beep sound as a placeholder
        sample_rate = 44100
        duration = 3  # seconds
        
        # Generate a simple sine wave
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        tone = np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
        
        # Normalize to 16-bit range
        tone = tone * 32767 / np.max(np.abs(tone))
        tone = tone.astype(np.int16)
        
        # Convert to audio segment
        audio = AudioSegment(
            tone.tobytes(),
            frame_rate=sample_rate,
            sample_width=2,  # 16-bit
            channels=1
        )
        
        # Export as MP3
        audio.export(audio_file, format="mp3")
        
        logger.info(f"Audio file created: {audio_file}")
        return audio_file
