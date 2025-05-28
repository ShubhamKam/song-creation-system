"""
Main Streamlit application for the Song Creation System.

This file integrates all components of the song creation system into a Streamlit web application.
"""

import streamlit as st
import os
import sys
import time
import tempfile
from PIL import Image
import base64
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import json

# Import system modules
from youtube_extractor import YouTubeExtractor
from audio_analyzer import AudioAnalyzer
from song_generator import SongGenerator
from voice_customizer import VoiceCustomizer
from knowledge_base import KnowledgeBase

# Set page configuration
st.set_page_config(
    page_title="Song Creation System",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #E8F5E9;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .warning-box {
        background-color: #FFF8E1;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .error-box {
        background-color: #FFEBEE;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .stAudio {
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .stProgress {
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'current_step' not in st.session_state:
    st.session_state.current_step = 'input'
if 'youtube_url' not in st.session_state:
    st.session_state.youtube_url = ''
if 'extracted_audio' not in st.session_state:
    st.session_state.extracted_audio = None
if 'audio_metadata' not in st.session_state:
    st.session_state.audio_metadata = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'generated_song' not in st.session_state:
    st.session_state.generated_song = None
if 'generation_info' not in st.session_state:
    st.session_state.generation_info = None
if 'voice_sample' not in st.session_state:
    st.session_state.voice_sample = None
if 'customized_song' not in st.session_state:
    st.session_state.customized_song = None
if 'history' not in st.session_state:
    st.session_state.history = []
if 'temp_dir' not in st.session_state:
    st.session_state.temp_dir = tempfile.mkdtemp()
if 'song_id' not in st.session_state:
    st.session_state.song_id = None

# Initialize system components
@st.cache_resource
def get_youtube_extractor():
    return YouTubeExtractor(output_dir=st.session_state.temp_dir)

@st.cache_resource
def get_audio_analyzer():
    return AudioAnalyzer(output_dir=st.session_state.temp_dir)

@st.cache_resource
def get_song_generator():
    return SongGenerator(output_dir=st.session_state.temp_dir)

@st.cache_resource
def get_voice_customizer():
    return VoiceCustomizer(output_dir=st.session_state.temp_dir)

@st.cache_resource
def get_knowledge_base():
    return KnowledgeBase(data_dir=os.path.join(st.session_state.temp_dir, "knowledge_base"))

# Get system components
youtube_extractor = get_youtube_extractor()
audio_analyzer = get_audio_analyzer()
song_generator = get_song_generator()
voice_customizer = get_voice_customizer()
knowledge_base = get_knowledge_base()

# Sidebar navigation
st.sidebar.markdown("# Song Creation System 🎵")
st.sidebar.markdown("---")

# Main navigation
nav_option = st.sidebar.radio(
    "Navigation",
    ["Home", "YouTube Input", "Audio Analysis", "Song Generation", "Voice Customization", "History & Feedback"]
)

# Update current step based on navigation
if nav_option == "YouTube Input":
    st.session_state.current_step = 'input'
elif nav_option == "Audio Analysis":
    st.session_state.current_step = 'analysis'
elif nav_option == "Song Generation":
    st.session_state.current_step = 'generation'
elif nav_option == "Voice Customization":
    st.session_state.current_step = 'voice'
elif nav_option == "History & Feedback":
    st.session_state.current_step = 'history'

# Home page
if nav_option == "Home":
    st.markdown("<h1 class='main-header'>Welcome to the Song Creation System</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    This application allows you to create unique songs with your voice by analyzing existing songs from YouTube.
    
    ## How it works:
    
    1. **Input a YouTube link** - Provide a link to a song you like
    2. **Analyze the audio** - Extract musical features like rhythm, harmony, and structure
    3. **Generate a new song** - Create a unique composition based on the analysis
    4. **Customize with your voice** - Apply your voice characteristics to the generated song
    5. **Provide feedback** - Help improve future generations
    
    Get started by navigating to the "YouTube Input" section in the sidebar.
    """)
    
    # Display workflow diagram
    cols = st.columns(5)
    for i, step in enumerate(["YouTube Input", "Audio Analysis", "Song Generation", "Voice Customization", "Feedback"]):
        with cols[i]:
            st.markdown(f"### {i+1}. {step}")
            st.markdown("↓" if i < 4 else "")
    
    st.markdown("---")
    
    st.markdown("### Recent Updates")
    st.markdown("""
    - Added support for multiple generation algorithms
    - Improved voice customization quality
    - Enhanced feedback analysis for better recommendations
    """)

# YouTube Input page
elif nav_option == "YouTube Input":
    st.markdown("<h1 class='main-header'>YouTube Input</h1>", unsafe_allow_html=True)
    
    with st.form("youtube_form"):
        youtube_url = st.text_input("Enter YouTube URL", 
                                   value=st.session_state.youtube_url,
                                   placeholder="https://www.youtube.com/watch?v=...")
        
        col1, col2 = st.columns(2)
        with col1:
            audio_quality = st.select_slider(
                "Audio Quality",
                options=["Low", "Medium", "High"],
                value="High"
            )
        with col2:
            audio_format = st.selectbox(
                "Audio Format",
                options=["MP3", "WAV", "FLAC"],
                index=0
            )
            
        submitted = st.form_submit_button("Process YouTube Link")
        
        if submitted and youtube_url:
            # Validate YouTube URL
            if not youtube_extractor.validate_youtube_url(youtube_url):
                st.error("Invalid YouTube URL. Please enter a valid YouTube video URL.")
            else:
                st.session_state.youtube_url = youtube_url
                
                # Extract audio from YouTube
                with st.spinner("Extracting audio from YouTube..."):
                    progress_bar = st.progress(0)
                    
                    # Define progress callback
                    def update_progress(progress):
                        progress_bar.progress(progress)
                    
                    try:
                        # Extract audio
                        audio_file, metadata = youtube_extractor.extract_audio(
                            youtube_url, 
                            quality=audio_quality.lower(),
                            format=audio_format.lower(),
                            progress_callback=update_progress
                        )
                        
                        # Store results in session state
                        st.session_state.extracted_audio = audio_file
                        st.session_state.audio_metadata = metadata
                        
                        st.success("Audio extracted successfully!")
                        
                        # Display video thumbnail and info
                        st.markdown("### Video Information")
                        cols = st.columns([1, 2])
                        with cols[0]:
                            if metadata.get('thumbnail'):
                                st.image(metadata['thumbnail'], use_column_width=True)
                            else:
                                st.image("https://via.placeholder.com/320x180.png?text=No+Thumbnail", use_column_width=True)
                        with cols[1]:
                            st.markdown(f"**Title:** {metadata.get('title', 'Unknown')}")
                            st.markdown(f"**Channel:** {metadata.get('uploader', 'Unknown')}")
                            st.markdown(f"**Duration:** {metadata.get('duration', 0)} seconds")
                            st.markdown(f"**Published:** {metadata.get('upload_date', 'Unknown')}")
                    
                    except Exception as e:
                        st.error(f"Error extracting audio: {str(e)}")
    
    # Display extracted audio if available
    if st.session_state.extracted_audio and os.path.exists(st.session_state.extracted_audio):
        st.markdown("### Extracted Audio")
        
        # Display audio player
        with open(st.session_state.extracted_audio, "rb") as audio_file:
            audio_bytes = audio_file.read()
            file_extension = os.path.splitext(st.session_state.extracted_audio)[1].lower()
            audio_format = f"audio/{file_extension[1:]}"
            st.audio(audio_bytes, format=audio_format)
        
        # Proceed to analysis button
        if st.button("Proceed to Audio Analysis"):
            st.session_state.current_step = 'analysis'
            st.experimental_rerun()

# Audio Analysis page
elif nav_option == "Audio Analysis":
    st.markdown("<h1 class='main-header'>Audio Analysis</h1>", unsafe_allow_html=True)
    
    if not st.session_state.extracted_audio or not os.path.exists(st.session_state.extracted_audio):
        st.warning("No audio has been extracted yet. Please process a YouTube link first.")
        if st.button("Go to YouTube Input"):
            st.session_state.current_step = 'input'
            st.experimental_rerun()
    else:
        # Analyze button
        if not st.session_state.analysis_results:
            if st.button("Analyze Audio"):
                with st.spinner("Analyzing audio features..."):
                    progress_bar = st.progress(0)
                    
                    # Define progress callback
                    def update_progress(progress):
                        progress_bar.progress(progress)
                    
                    try:
                        # Analyze audio
                        analysis_results = audio_analyzer.analyze_audio(
                            st.session_state.extracted_audio,
                            progress_callback=update_progress
                        )
                        
                        # Store results in session state
                        st.session_state.analysis_results = analysis_results
                        
                        # Add to knowledge base
                        st.session_state.song_id = knowledge_base.add_analyzed_song(
                            st.session_state.youtube_url,
                            analysis_results
                        )
                        
                        st.success("Audio analysis completed successfully!")
                        st.experimental_rerun()
                        
                    except Exception as e:
                        st.error(f"Error analyzing audio: {str(e)}")
        
        # Display analysis results if available
        if st.session_state.analysis_results:
            # Tabs for different analysis types
            analysis_tabs = st.tabs(["Rhythm", "Harmony", "Timbre", "Structure", "Lyrics"])
            
            with analysis_tabs[0]:  # Rhythm
                st.markdown("### Rhythm Analysis")
                
                # Display rhythm metrics
                rhythm_data = st.session_state.analysis_results.get('rhythm', {})
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Tempo (BPM)", f"{rhythm_data.get('tempo', 0):.1f}")
                with col2:
                    st.metric("Beat Strength", f"{rhythm_data.get('beat_strength', 0):.2f}")
                with col3:
                    st.metric("Rhythm Regularity", f"{rhythm_data.get('rhythm_regularity', 0):.2f}")
                
                # Display rhythm visualization
                st.markdown("#### Beat Pattern")
                beat_times = rhythm_data.get('beat_times', [])
                if beat_times:
                    # Create beat visualization
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.vlines(beat_times, 0, 1, color='r', alpha=0.5, linewidth=1)
                    ax.set_xlabel('Time (s)')
                    ax.set_ylabel('Beat')
                    ax.set_title('Beat Locations')
                    ax.set_yticks([])
                    ax.set_xlim(0, min(60, beat_times[-1] if beat_times else 30))  # Show first minute max
                    
                    # Display plot
                    st.pyplot(fig)
                else:
                    st.info("No beat pattern data available.")
            
            with analysis_tabs[1]:  # Harmony
                st.markdown("### Harmony Analysis")
                
                # Display harmony metrics
                harmony_data = st.session_state.analysis_results.get('harmony', {})
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Key", harmony_data.get('key', 'Unknown'))
                    st.metric("Mode", harmony_data.get('mode', 'Unknown'))
                with col2:
                    st.metric("Chord Complexity", f"{harmony_data.get('chord_complexity', 0):.2f}")
                    st.metric("Harmonic Novelty", f"{harmony_data.get('harmonic_novelty', 0):.2f}")
                
                # Display chroma visualization
                st.markdown("#### Chromagram")
                chroma = harmony_data.get('chroma', [])
                if chroma:
                    # Convert to numpy array if it's a list
                    if isinstance(chroma, list):
                        chroma = np.array(chroma)
                    
                    # Create chromagram visualization
                    fig, ax = plt.subplots(figsize=(10, 4))
                    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
                    
                    if len(chroma.shape) > 1:
                        # Full chromagram
                        im = ax.imshow(chroma, aspect='auto', origin='lower', cmap='viridis')
                        ax.set_yticks(np.arange(12))
                        ax.set_yticklabels(notes)
                        ax.set_xlabel('Time (frames)')
                        ax.set_ylabel('Pitch Class')
                        plt.colorbar(im, ax=ax)
                    else:
                        # Average chroma
                        ax.bar(notes, chroma)
                        ax.set_ylabel('Magnitude')
                        ax.set_title('Average Chroma')
                    
                    # Display plot
                    st.pyplot(fig)
                else:
                    st.info("No chromagram data available.")
            
            with analysis_tabs[2]:  # Timbre
                st.markdown("### Timbre Analysis")
                
                # Display timbre visualization
                st.markdown("#### Spectral Features")
                timbre_data = st.session_state.analysis_results.get('timbre', {})
                
                # Display MFCCs
                mfccs_mean = timbre_data.get('mfccs_mean', [])
                if mfccs_mean:
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.bar(range(len(mfccs_mean)), mfccs_mean)
                    ax.set_xlabel('MFCC Coefficient')
                    ax.set_ylabel('Magnitude')
                    ax.set_title('MFCC Features')
                    st.pyplot(fig)
                
                # Display instrument detection
                st.markdown("#### Detected Instruments")
                instruments = timbre_data.get('instruments', {})
                
                if instruments:
                    # Sort instruments by confidence
                    sorted_instruments = dict(sorted(instruments.items(), key=lambda x: x[1], reverse=True))
                    
                    # Display as horizontal bars
                    for instrument, confidence in sorted_instruments.items():
                        st.markdown(f"{instrument}: {confidence:.2f}")
                        st.progress(confidence)
                else:
                    st.info("No instrument detection data available.")
            
            with analysis_tabs[3]:  # Structure
                st.markdown("### Song Structure Analysis")
                
                # Display song structure
                st.markdown("#### Song Sections")
                structure_data = st.session_state.analysis_results.get('structure', {})
                sections = structure_data.get('sections', [])
                
                if sections:
                    # Convert to DataFrame for display
                    sections_df = pd.DataFrame(sections)
                    st.dataframe(sections_df, use_container_width=True)
                    
                    # Display structure visualization
                    st.markdown("#### Visual Structure")
                    
                    # Create a simple visualization of song structure
                    fig, ax = plt.subplots(figsize=(10, 4))
                    
                    # Define colors for different section types
                    colors = {
                        'Intro': '#1E88E5',
                        'Verse': '#43A047',
                        'Chorus': '#FDD835',
                        'Bridge': '#FB8C00',
                        'Outro': '#757575'
                    }
                    
                    # Plot sections as colored blocks
                    for section in sections:
                        section_type = section['section'].split()[0] if ' ' in section['section'] else section['section']
                        color = colors.get(section_type, '#757575')
                        ax.barh(0, section['duration'], left=section['start'], color=color, alpha=0.7)
                        
                        # Add text label if section is wide enough
                        if section['duration'] > 5:
                            ax.text(
                                section['start'] + section['duration']/2, 
                                0, 
                                section['section'],
                                ha='center', 
                                va='center', 
                                color='black',
                                fontweight='bold'
                            )
                    
                    # Set plot properties
                    ax.set_yticks([])
                    ax.set_xlabel('Time (seconds)')
                    ax.set_title('Song Structure')
                    
                    # Add legend
                    from matplotlib.patches import Patch
                    legend_elements = [Patch(facecolor=color, label=section)
                                      for section, color in colors.items()]
                    ax.legend(handles=legend_elements, loc='upper center', 
                             bbox_to_anchor=(0.5, -0.15), ncol=5)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.info("No song structure data available.")
            
            with analysis_tabs[4]:  # Lyrics
                st.markdown("### Lyric Analysis")
                
                st.info("Lyric transcription and analysis is not implemented in this demo version. In a full implementation, this would use speech recognition to transcribe lyrics and NLP to analyze themes, sentiment, and structure.")
                
                # Placeholder for lyric transcription
                st.markdown("#### Transcribed Lyrics")
                st.text_area("Lyrics", "Lyrics would be transcribed here in the full version.", height=300, disabled=True)
        
        # Proceed to generation button
        if st.session_state.analysis_results:
            if st.button("Proceed to Song Generation"):
                st.session_state.current_step = 'generation'
                st.experimental_rerun()

# Song Generation page
elif nav_option == "Song Generation":
    st.markdown("<h1 class='main-header'>Song Generation</h1>", unsafe_allow_html=True)
    
    if not st.session_state.analysis_results:
        st.warning("No audio analysis available. Please analyze a song first.")
        if st.button("Go to Audio Analysis"):
            st.session_state.current_step = 'analysis'
            st.experimental_rerun()
    else:
        # Generation parameters
        st.markdown("### Generation Parameters")
        
        col1, col2 = st.columns(2)
        with col1:
            generation_model = st.selectbox(
                "Generation Model",
                options=["Neural Network", "Markov Chain", "Transformer", "Hybrid"],
                index=0
            )
            
            creativity = st.slider(
                "Creativity Level",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.1,
                help="Higher values produce more creative but potentially less coherent results"
            )
        
        with col2:
            similarity = st.slider(
                "Similarity to Original",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.1,
                help="Higher values produce results more similar to the original song"
            )
            
            duration = st.slider(
                "Duration (seconds)",
                min_value=30,
                max_value=300,
                value=180,
                step=30
            )
        
        # Advanced parameters in expander
        with st.expander("Advanced Parameters"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                tempo_variation = st.slider(
                    "Tempo Variation",
                    min_value=-20,
                    max_value=20,
                    value=0,
                    step=5,
                    help="Adjust the tempo relative to the original"
                )
                
                key_shift = st.slider(
                    "Key Shift",
                    min_value=-6,
                    max_value=6,
                    value=0,
                    step=1,
                    help="Shift the key up or down by semitones"
                )
            
            with col2:
                rhythm_complexity = st.slider(
                    "Rhythm Complexity",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.1
                )
                
                harmonic_complexity = st.slider(
                    "Harmonic Complexity",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.1
                )
            
            with col3:
                structure_variation = st.slider(
                    "Structure Variation",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.3,
                    step=0.1
                )
                
                lyric_influence = st.slider(
                    "Lyric Influence",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.6,
                    step=0.1
                )
        
        # Generate button
        if st.button("Generate Song"):
            with st.spinner("Generating song..."):
                progress_bar = st.progress(0)
                
                # Define progress callback
                def update_progress(progress):
                    progress_bar.progress(progress)
                
                try:
                    # Prepare generation parameters
                    params = {
                        'model': generation_model.lower().replace(' ', '_'),
                        'creativity': creativity,
                        'similarity': similarity,
                        'duration': duration,
                        'tempo_variation': tempo_variation,
                        'key_shift': key_shift,
                        'rhythm_complexity': rhythm_complexity,
                        'harmonic_complexity': harmonic_complexity,
                        'structure_variation': structure_variation,
                        'lyric_influence': lyric_influence
                    }
                    
                    # Generate song
                    generation_info = song_generator.generate_song(
                        st.session_state.analysis_results,
                        params=params,
                        progress_callback=update_progress
                    )
                    
                    # Store results in session state
                    st.session_state.generated_song = generation_info['audio_file']
                    st.session_state.generation_info = generation_info
                    
                    st.success("Song generated successfully!")
                    st.experimental_rerun()
                    
                except Exception as e:
                    st.error(f"Error generating song: {str(e)}")
        
        # Display generated song if available
        if st.session_state.generated_song and os.path.exists(st.session_state.generated_song):
            st.markdown("### Generated Song")
            
            # Display audio player
            with open(st.session_state.generated_song, "rb") as audio_file:
                audio_bytes = audio_file.read()
                file_extension = os.path.splitext(st.session_state.generated_song)[1].lower()
                audio_format = f"audio/{file_extension[1:]}"
                st.audio(audio_bytes, format=audio_format)
            
            # Display generation info
            if st.session_state.generation_info:
                st.markdown("### Generation Information")
                info = st.session_state.generation_info
                
                # Create a more readable version of the info
                display_info = {
                    "Model": info.get('model', 'Unknown'),
                    "Creativity": info.get('creativity', 0),
                    "Similarity": info.get('similarity', 0),
                    "Duration": f"{info.get('duration', 0)} seconds",
                    "Tempo": f"{info.get('tempo', 0)} BPM",
                    "Key": info.get('key', 'Unknown'),
                    "Mode": info.get('mode', 'Unknown'),
                    "Timestamp": info.get('timestamp', 'Unknown')
                }
                
                # Display as JSON
                st.json(display_info)
            
            # Download button
            if os.path.exists(st.session_state.generated_song):
                with open(st.session_state.generated_song, "rb") as file:
                    st.download_button(
                        label="Download Generated Song",
                        data=file,
                        file_name=os.path.basename(st.session_state.generated_song),
                        mime=f"audio/{os.path.splitext(st.session_state.generated_song)[1][1:]}"
                    )
            
            # Proceed to voice customization button
            if st.button("Proceed to Voice Customization"):
                st.session_state.current_step = 'voice'
                st.experimental_rerun()

# Voice Customization page
elif nav_option == "Voice Customization":
    st.markdown("<h1 class='main-header'>Voice Customization</h1>", unsafe_allow_html=True)
    
    if not st.session_state.generated_song or not os.path.exists(st.session_state.generated_song):
        st.warning("No generated song available. Please generate a song first.")
        if st.button("Go to Song Generation"):
            st.session_state.current_step = 'generation'
            st.experimental_rerun()
    else:
        # Voice sample upload
        st.markdown("### Upload Voice Sample")
        
        uploaded_file = st.file_uploader(
            "Upload a recording of your voice (MP3, WAV, or M4A)",
            type=["mp3", "wav", "m4a"]
        )
        
        if uploaded_file:
            # Save the uploaded file
            voice_sample_path = os.path.join(st.session_state.temp_dir, uploaded_file.name)
            with open(voice_sample_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.session_state.voice_sample = voice_sample_path
            
            # Display voice sample
            st.markdown("#### Voice Sample Preview")
            st.audio(uploaded_file, format=f"audio/{uploaded_file.name.split('.')[-1]}")
        
        # Voice customization parameters
        if st.session_state.voice_sample and os.path.exists(st.session_state.voice_sample):
            st.markdown("### Voice Customization Parameters")
            
            col1, col2 = st.columns(2)
            with col1:
                voice_pitch = st.slider(
                    "Voice Pitch",
                    min_value=-12,
                    max_value=12,
                    value=0,
                    step=1,
                    help="Adjust the pitch of your voice in semitones"
                )
                
                voice_timbre = st.slider(
                    "Voice Timbre",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.7,
                    step=0.1,
                    help="How much of your voice's timbre to preserve"
                )
            
            with col2:
                voice_vibrato = st.slider(
                    "Voice Vibrato",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.1
                )
                
                voice_clarity = st.slider(
                    "Voice Clarity",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.8,
                    step=0.1
                )
            
            # Apply voice button
            if st.button("Apply Voice to Song"):
                with st.spinner("Applying voice customization..."):
                    progress_bar = st.progress(0)
                    
                    # Define progress callback
                    def update_progress(progress):
                        progress_bar.progress(progress)
                    
                    try:
                        # Prepare voice parameters
                        params = {
                            'voice_pitch': voice_pitch,
                            'voice_timbre': voice_timbre,
                            'voice_vibrato': voice_vibrato,
                            'voice_clarity': voice_clarity
                        }
                        
                        # Apply voice customization
                        customization_result = voice_customizer.customize_voice(
                            st.session_state.voice_sample,
                            st.session_state.generated_song,
                            params=params,
                            progress_callback=update_progress
                        )
                        
                        # Store results in session state
                        st.session_state.customized_song = customization_result['customized_song']
                        
                        st.success("Voice customization applied successfully!")
                        st.experimental_rerun()
                        
                    except Exception as e:
                        st.error(f"Error applying voice customization: {str(e)}")
            
            # Display customized song if available
            if st.session_state.customized_song and os.path.exists(st.session_state.customized_song):
                st.markdown("### Customized Song")
                
                # Display audio player
                with open(st.session_state.customized_song, "rb") as audio_file:
                    audio_bytes = audio_file.read()
                    file_extension = os.path.splitext(st.session_state.customized_song)[1].lower()
                    audio_format = f"audio/{file_extension[1:]}"
                    st.audio(audio_bytes, format=audio_format)
                
                # Download button
                with open(st.session_state.customized_song, "rb") as file:
                    st.download_button(
                        label="Download Customized Song",
                        data=file,
                        file_name=os.path.basename(st.session_state.customized_song),
                        mime=f"audio/{os.path.splitext(st.session_state.customized_song)[1][1:]}"
                    )
                
                # Save to history
                if st.button("Save to History and Provide Feedback"):
                    # Add to history
                    if st.session_state.song_id and st.session_state.generation_info:
                        # Add to session state history for display
                        st.session_state.history.append({
                            "youtube_url": st.session_state.youtube_url,
                            "generation_model": st.session_state.generation_info.get('model', 'unknown'),
                            "voice_customization": True,
                            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                            "rating": None,
                            "feedback": None
                        })
                    
                    st.session_state.current_step = 'history'
                    st.experimental_rerun()

# History & Feedback page
elif nav_option == "History & Feedback":
    st.markdown("<h1 class='main-header'>History & Feedback</h1>", unsafe_allow_html=True)
    
    # Get history from knowledge base
    kb_history = knowledge_base.get_song_history()
    
    if not kb_history and not st.session_state.history:
        st.info("No song creation history available yet. Create your first song to see it here.")
    else:
        # Display history
        st.markdown("### Your Song Creation History")
        
        # Combine knowledge base history with session history
        display_history = kb_history if kb_history else st.session_state.history
        
        for i, item in enumerate(display_history):
            # For knowledge base items
            if 'youtube_url' in item:
                youtube_url = item['youtube_url']
                timestamp = item.get('date_added', 'Unknown')
                features = item.get('features', {})
                feedback_list = item.get('feedback', [])
                
                with st.expander(f"Song {i+1} - {timestamp}"):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown(f"**YouTube Source:** {youtube_url}")
                        if features:
                            st.markdown(f"**Tempo:** {features.get('tempo', 'Unknown')} BPM")
                            st.markdown(f"**Key:** {features.get('key', 'Unknown')} {features.get('mode', '')}")
                        
                        # If we have audio files, display them
                        if st.session_state.extracted_audio and os.path.exists(st.session_state.extracted_audio):
                            st.markdown("**Original Audio:**")
                            with open(st.session_state.extracted_audio, "rb") as audio_file:
                                audio_bytes = audio_file.read()
                                file_extension = os.path.splitext(st.session_state.extracted_audio)[1].lower()
                                audio_format = f"audio/{file_extension[1:]}"
                                st.audio(audio_bytes, format=audio_format)
                        
                        if st.session_state.customized_song and os.path.exists(st.session_state.customized_song):
                            st.markdown("**Customized Song:**")
                            with open(st.session_state.customized_song, "rb") as audio_file:
                                audio_bytes = audio_file.read()
                                file_extension = os.path.splitext(st.session_state.customized_song)[1].lower()
                                audio_format = f"audio/{file_extension[1:]}"
                                st.audio(audio_bytes, format=audio_format)
                    
                    with col2:
                        # Rating widget
                        st.markdown("**Rate this song:**")
                        
                        # Create a custom star rating widget
                        rating_cols = st.columns(5)
                        
                        # Get current rating
                        current_rating = None
                        if feedback_list:
                            # Use the most recent feedback
                            current_rating = feedback_list[-1].get('rating', None)
                        
                        rating = current_rating or 0
                        
                        # Display stars
                        for star in range(1, 6):
                            with rating_cols[star-1]:
                                if st.button(
                                    "★" if star <= rating else "☆", 
                                    key=f"star_{i}_{star}"
                                ):
                                    # Update rating
                                    rating = star
                                    # In a real implementation, this would update the knowledge base
                        
                        # Feedback text area
                        feedback_text = ""
                        if feedback_list:
                            # Use the most recent feedback
                            feedback_text = feedback_list[-1].get('feedback_text', "")
                        
                        new_feedback = st.text_area(
                            "Feedback",
                            value=feedback_text,
                            key=f"feedback_{i}"
                        )
                        
                        # Submit feedback button
                        if st.button("Submit Feedback", key=f"submit_{i}"):
                            if st.session_state.song_id and st.session_state.generation_info:
                                try:
                                    # Add feedback to knowledge base
                                    feedback_id = knowledge_base.add_generation_feedback(
                                        st.session_state.song_id,
                                        st.session_state.generation_info,
                                        rating,
                                        new_feedback
                                    )
                                    
                                    # Analyze feedback with LLM
                                    if new_feedback:
                                        feedback_analysis = knowledge_base.analyze_feedback_with_llm(new_feedback)
                                        
                                        # Display analysis
                                        st.markdown("### Feedback Analysis")
                                        st.markdown(f"**Overall Sentiment:** {feedback_analysis['overall_sentiment'].capitalize()}")
                                        
                                        if feedback_analysis['suggestions']:
                                            st.markdown("**Suggestions for Improvement:**")
                                            for suggestion in feedback_analysis['suggestions']:
                                                st.markdown(f"- {suggestion}")
                                    
                                    st.success("Feedback submitted successfully!")
                                    
                                except Exception as e:
                                    st.error(f"Error submitting feedback: {str(e)}")
        
        # Recommendations based on history
        st.markdown("### Recommendations")
        st.markdown("Based on your history, you might enjoy songs created with these parameters:")
        
        # Get recommendations from knowledge base
        recommendations = knowledge_base.get_recommendations(3)
        
        if recommendations:
            rec_cols = st.columns(len(recommendations))
            
            for i, rec in enumerate(recommendations):
                with rec_cols[i]:
                    st.markdown(f"#### Recommendation {i+1}")
                    st.markdown(f"**Style:** {rec.get('style', 'Unknown')}")
                    
                    tempo_range = rec.get('tempo_range', [100, 130])
                    st.markdown(f"**Tempo:** {tempo_range[0]}-{tempo_range[1]} BPM")
                    
                    st.markdown(f"**Key:** {rec.get('key', 'C Major')}")
                    st.markdown(f"**Complexity:** {rec.get('complexity', 0.5):.1f}")
                    
                    if st.button("Generate with these parameters", key=f"rec_{i}"):
                        st.info("In a full implementation, this would generate a new song with the recommended parameters.")
        else:
            st.info("Create more songs and provide feedback to get personalized recommendations.")

# Footer
st.markdown("---")
st.markdown("© 2025 Song Creation System | Created with Streamlit")
