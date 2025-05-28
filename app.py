import streamlit as st
import os
import sys
import time
import tempfile
from PIL import Image
import base64

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
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'generated_song' not in st.session_state:
    st.session_state.generated_song = None
if 'voice_sample' not in st.session_state:
    st.session_state.voice_sample = None
if 'customized_song' not in st.session_state:
    st.session_state.customized_song = None
if 'history' not in st.session_state:
    st.session_state.history = []

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
            st.session_state.youtube_url = youtube_url
            
            # Simulate extraction process
            with st.spinner("Extracting audio from YouTube..."):
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)  # Simulate work being done
                    progress_bar.progress(i + 1)
                
                # In a real implementation, we would use yt-dlp here
                st.session_state.extracted_audio = "dummy_audio.mp3"  # Placeholder
                
                st.success("Audio extracted successfully!")
                
                # Display video thumbnail and info (simulated)
                st.markdown("### Video Information")
                cols = st.columns([1, 2])
                with cols[0]:
                    st.image("https://via.placeholder.com/320x180.png?text=Video+Thumbnail", use_column_width=True)
                with cols[1]:
                    st.markdown("**Title:** Sample YouTube Video")
                    st.markdown("**Channel:** Sample Channel")
                    st.markdown("**Duration:** 3:45")
                    st.markdown("**Published:** January 1, 2025")
    
    # Display extracted audio if available
    if st.session_state.extracted_audio:
        st.markdown("### Extracted Audio")
        
        # In a real implementation, we would use the actual audio file
        # For now, use a sample audio file
        sample_audio = open("sample_audio.mp3", "rb").read() if os.path.exists("sample_audio.mp3") else None
        
        if sample_audio:
            st.audio(sample_audio, format="audio/mp3")
        else:
            st.markdown("*Audio preview not available in this demo*")
        
        # Proceed to analysis button
        if st.button("Proceed to Audio Analysis"):
            st.session_state.current_step = 'analysis'
            st.experimental_rerun()

# Audio Analysis page
elif nav_option == "Audio Analysis":
    st.markdown("<h1 class='main-header'>Audio Analysis</h1>", unsafe_allow_html=True)
    
    if not st.session_state.extracted_audio:
        st.warning("No audio has been extracted yet. Please process a YouTube link first.")
        if st.button("Go to YouTube Input"):
            st.session_state.current_step = 'input'
            st.experimental_rerun()
    else:
        # Tabs for different analysis types
        analysis_tabs = st.tabs(["Rhythm", "Harmony", "Timbre", "Structure", "Lyrics"])
        
        with analysis_tabs[0]:  # Rhythm
            st.markdown("### Rhythm Analysis")
            
            # Simulate analysis process if not already done
            if not st.session_state.analysis_results:
                with st.spinner("Analyzing rhythm features..."):
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.01)  # Simulate work being done
                        progress_bar.progress(i + 1)
                    
                    # In a real implementation, we would use librosa here
                    st.session_state.analysis_results = {
                        "tempo": 120,
                        "beat_strength": 0.8,
                        "rhythm_regularity": 0.9
                    }
            
            # Display rhythm metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Tempo (BPM)", st.session_state.analysis_results["tempo"])
            with col2:
                st.metric("Beat Strength", f"{st.session_state.analysis_results['beat_strength']:.2f}")
            with col3:
                st.metric("Rhythm Regularity", f"{st.session_state.analysis_results['rhythm_regularity']:.2f}")
            
            # Display rhythm visualization (simulated)
            st.markdown("#### Beat Pattern")
            st.line_chart({"Beat Strength": [0.8, 0.2, 0.7, 0.3, 0.9, 0.1, 0.6, 0.4] * 4})
        
        with analysis_tabs[1]:  # Harmony
            st.markdown("### Harmony Analysis")
            
            # Display harmony metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Key", "C Major")
                st.metric("Mode Confidence", "0.85")
            with col2:
                st.metric("Chord Complexity", "Medium")
                st.metric("Harmonic Novelty", "0.72")
            
            # Display chord progression (simulated)
            st.markdown("#### Chord Progression")
            st.json({
                "intro": ["C", "Am", "F", "G"],
                "verse": ["C", "F", "Am", "G", "C", "F", "G"],
                "chorus": ["F", "G", "C", "Am", "F", "G", "C"],
                "outro": ["C", "G", "F", "C"]
            })
            
            # Display chord visualization (simulated)
            st.markdown("#### Chord Distribution")
            st.bar_chart({
                "C": 8, "Am": 4, "F": 6, "G": 6, "Dm": 2, "Em": 1
            })
        
        with analysis_tabs[2]:  # Timbre
            st.markdown("### Timbre Analysis")
            
            # Display timbre visualization (simulated)
            st.markdown("#### Spectral Features")
            st.line_chart({
                "Spectral Centroid": [0.5, 0.52, 0.48, 0.55, 0.6, 0.58, 0.54, 0.52] * 4,
                "Spectral Bandwidth": [0.3, 0.32, 0.28, 0.35, 0.4, 0.38, 0.34, 0.32] * 4
            })
            
            # Display instrument detection (simulated)
            st.markdown("#### Detected Instruments")
            instruments = {
                "Guitar": 0.95,
                "Piano": 0.85,
                "Drums": 0.98,
                "Bass": 0.92,
                "Vocals": 0.99,
                "Strings": 0.45,
                "Synth": 0.75
            }
            
            # Sort instruments by confidence
            sorted_instruments = dict(sorted(instruments.items(), key=lambda x: x[1], reverse=True))
            
            # Display as horizontal bars
            for instrument, confidence in sorted_instruments.items():
                st.markdown(f"{instrument}: {confidence:.2f}")
                st.progress(confidence)
        
        with analysis_tabs[3]:  # Structure
            st.markdown("### Song Structure Analysis")
            
            # Display song structure (simulated)
            st.markdown("#### Song Sections")
            structure = [
                {"section": "Intro", "start": "0:00", "end": "0:15", "duration": "15s"},
                {"section": "Verse 1", "start": "0:15", "end": "0:45", "duration": "30s"},
                {"section": "Chorus", "start": "0:45", "end": "1:15", "duration": "30s"},
                {"section": "Verse 2", "start": "1:15", "end": "1:45", "duration": "30s"},
                {"section": "Chorus", "start": "1:45", "end": "2:15", "duration": "30s"},
                {"section": "Bridge", "start": "2:15", "end": "2:45", "duration": "30s"},
                {"section": "Chorus", "start": "2:45", "end": "3:15", "duration": "30s"},
                {"section": "Outro", "start": "3:15", "end": "3:30", "duration": "15s"}
            ]
            
            st.dataframe(structure, use_container_width=True)
            
            # Display structure visualization (simulated)
            st.markdown("#### Visual Structure")
            
            # Create a simple visualization of song structure
            cols = st.columns(8)
            for i, section in enumerate(structure):
                with cols[i]:
                    color = "#1E88E5" if section["section"] == "Chorus" else (
                        "#43A047" if section["section"].startswith("Verse") else (
                            "#FDD835" if section["section"] == "Bridge" else "#757575"
                        )
                    )
                    st.markdown(
                        f"""
                        <div style="background-color: {color}; 
                                    height: {int(section['duration'][:-1]) * 3}px; 
                                    border-radius: 5px; 
                                    padding: 5px; 
                                    color: white; 
                                    text-align: center;">
                            {section['section']}
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
        
        with analysis_tabs[4]:  # Lyrics
            st.markdown("### Lyric Analysis")
            
            # Display transcribed lyrics (simulated)
            st.markdown("#### Transcribed Lyrics")
            lyrics = """
            Verse 1:
            This is a sample song
            With some example lyrics
            Just to demonstrate
            How the system works
            
            Chorus:
            This is the chorus part
            The most memorable section
            Where the hook resides
            And emotions peak
            
            Verse 2:
            Second verse continues
            With the song's narrative
            Building on the theme
            Established before
            
            Chorus:
            This is the chorus part
            The most memorable section
            Where the hook resides
            And emotions peak
            
            Bridge:
            Here's a contrasting section
            Different from the rest
            Providing some variety
            Before the final chorus
            
            Chorus:
            This is the chorus part
            The most memorable section
            Where the hook resides
            And emotions peak
            """
            
            st.text_area("Lyrics", lyrics, height=300)
            
            # Display lyric analysis (simulated)
            st.markdown("#### Lyric Analysis")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Sentiment", "Positive (0.68)")
                st.metric("Complexity", "Medium (0.55)")
            with col2:
                st.metric("Repetition", "High (0.82)")
                st.metric("Rhyme Density", "Medium (0.61)")
            
            # Display theme analysis (simulated)
            st.markdown("#### Theme Analysis")
            themes = {
                "Love": 0.25,
                "Hope": 0.65,
                "Journey": 0.45,
                "Reflection": 0.55,
                "Struggle": 0.15,
                "Celebration": 0.40
            }
            
            # Sort themes by relevance
            sorted_themes = dict(sorted(themes.items(), key=lambda x: x[1], reverse=True))
            
            # Display as horizontal bars
            for theme, relevance in sorted_themes.items():
                st.markdown(f"{theme}: {relevance:.2f}")
                st.progress(relevance)
        
        # Proceed to generation button
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
                for i in range(100):
                    time.sleep(0.03)  # Simulate work being done
                    progress_bar.progress(i + 1)
                
                # In a real implementation, we would use the generation model here
                st.session_state.generated_song = "dummy_generated.mp3"  # Placeholder
                
                st.success("Song generated successfully!")
        
        # Display generated song if available
        if st.session_state.generated_song:
            st.markdown("### Generated Song")
            
            # In a real implementation, we would use the actual audio file
            # For now, use a sample audio file
            sample_audio = open("sample_audio.mp3", "rb").read() if os.path.exists("sample_audio.mp3") else None
            
            if sample_audio:
                st.audio(sample_audio, format="audio/mp3")
            else:
                st.markdown("*Audio preview not available in this demo*")
            
            # Display generation info
            st.markdown("### Generation Information")
            st.json({
                "model": generation_model,
                "creativity": creativity,
                "similarity": similarity,
                "duration": f"{duration} seconds",
                "tempo": f"{st.session_state.analysis_results['tempo'] + tempo_variation} BPM",
                "key": "C Major" if key_shift == 0 else f"{'C#' if key_shift > 0 else 'B'} Major",
                "timestamp": "2025-05-28 00:39:03"
            })
            
            # Download button
            st.download_button(
                label="Download Generated Song",
                data=sample_audio if sample_audio else b"",
                file_name="generated_song.mp3",
                mime="audio/mp3",
                disabled=not sample_audio
            )
            
            # Proceed to voice customization button
            if st.button("Proceed to Voice Customization"):
                st.session_state.current_step = 'voice'
                st.experimental_rerun()

# Voice Customization page
elif nav_option == "Voice Customization":
    st.markdown("<h1 class='main-header'>Voice Customization</h1>", unsafe_allow_html=True)
    
    if not st.session_state.generated_song:
        st.warning("No generated song available. Please generate a song first.")
        if st.button("Go to Song Generation"):
            st.session_state.current_step = 'generation'
            st.experimental_rerun()
    else:
        # Voice sample upload
        st.markdown("### Upload Voice Sample")
        
        voice_sample = st.file_uploader(
            "Upload a recording of your voice (MP3, WAV, or M4A)",
            type=["mp3", "wav", "m4a"]
        )
        
        if voice_sample:
            st.session_state.voice_sample = voice_sample
            
            # Display voice sample
            st.markdown("#### Voice Sample Preview")
            
            # In a real implementation, we would use the actual audio file
            # For now, use a sample audio file
            sample_audio = open("sample_audio.mp3", "rb").read() if os.path.exists("sample_audio.mp3") else None
            
            if sample_audio:
                st.audio(sample_audio, format="audio/mp3")
            else:
                st.markdown("*Audio preview not available in this demo*")
        
        # Voice customization parameters
        if st.session_state.voice_sample:
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
                    for i in range(100):
                        time.sleep(0.03)  # Simulate work being done
                        progress_bar.progress(i + 1)
                    
                    # In a real implementation, we would use voice synthesis here
                    st.session_state.customized_song = "dummy_customized.mp3"  # Placeholder
                    
                    st.success("Voice customization applied successfully!")
            
            # Display customized song if available
            if st.session_state.customized_song:
                st.markdown("### Customized Song")
                
                # In a real implementation, we would use the actual audio file
                # For now, use a sample audio file
                sample_audio = open("sample_audio.mp3", "rb").read() if os.path.exists("sample_audio.mp3") else None
                
                if sample_audio:
                    st.audio(sample_audio, format="audio/mp3")
                else:
                    st.markdown("*Audio preview not available in this demo*")
                
                # Download button
                st.download_button(
                    label="Download Customized Song",
                    data=sample_audio if sample_audio else b"",
                    file_name="customized_song.mp3",
                    mime="audio/mp3",
                    disabled=not sample_audio
                )
                
                # Save to history
                if st.button("Save to History and Provide Feedback"):
                    # Add to history
                    st.session_state.history.append({
                        "youtube_url": st.session_state.youtube_url,
                        "generation_model": "Neural Network",  # Example value
                        "voice_customization": True,
                        "timestamp": "2025-05-28 00:39:03",
                        "rating": None,
                        "feedback": None
                    })
                    
                    st.session_state.current_step = 'history'
                    st.experimental_rerun()

# History & Feedback page
elif nav_option == "History & Feedback":
    st.markdown("<h1 class='main-header'>History & Feedback</h1>", unsafe_allow_html=True)
    
    if not st.session_state.history:
        st.info("No song creation history available yet. Create your first song to see it here.")
    else:
        # Display history
        st.markdown("### Your Song Creation History")
        
        for i, item in enumerate(st.session_state.history):
            with st.expander(f"Song {i+1} - {item['timestamp']}"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**YouTube Source:** {item['youtube_url']}")
                    st.markdown(f"**Generation Model:** {item['generation_model']}")
                    st.markdown(f"**Voice Customization:** {'Yes' if item['voice_customization'] else 'No'}")
                    
                    # In a real implementation, we would use the actual audio file
                    # For now, use a sample audio file
                    sample_audio = open("sample_audio.mp3", "rb").read() if os.path.exists("sample_audio.mp3") else None
                    
                    if sample_audio:
                        st.audio(sample_audio, format="audio/mp3")
                    else:
                        st.markdown("*Audio preview not available in this demo*")
                
                with col2:
                    # Rating widget
                    st.markdown("**Rate this song:**")
                    
                    # Create a custom star rating widget
                    rating_cols = st.columns(5)
                    rating = item['rating'] or 0
                    
                    for star in range(1, 6):
                        with rating_cols[star-1]:
                            if st.button(
                                "★" if star <= rating else "☆", 
                                key=f"star_{i}_{star}"
                            ):
                                st.session_state.history[i]['rating'] = star
                                st.experimental_rerun()
                    
                    # Feedback text area
                    feedback = st.text_area(
                        "Feedback",
                        value=item['feedback'] or "",
                        key=f"feedback_{i}"
                    )
                    
                    if feedback != item['feedback']:
                        st.session_state.history[i]['feedback'] = feedback
                    
                    # Submit feedback button
                    if st.button("Submit Feedback", key=f"submit_{i}"):
                        st.success("Feedback submitted successfully!")
        
        # Recommendations based on history
        st.markdown("### Recommendations")
        st.markdown("Based on your history, you might enjoy songs created with these parameters:")
        
        # Display recommendations (simulated)
        rec_cols = st.columns(3)
        
        with rec_cols[0]:
            st.markdown("#### Recommendation 1")
            st.markdown("**Style:** Pop/Rock")
            st.markdown("**Tempo:** 120-130 BPM")
            st.markdown("**Key:** C Major")
            st.markdown("**Voice:** Higher pitch")
            
            if st.button("Generate with these parameters", key="rec_1"):
                st.info("This would generate a new song with the recommended parameters")
        
        with rec_cols[1]:
            st.markdown("#### Recommendation 2")
            st.markdown("**Style:** Electronic")
            st.markdown("**Tempo:** 140-150 BPM")
            st.markdown("**Key:** A Minor")
            st.markdown("**Voice:** More vibrato")
            
            if st.button("Generate with these parameters", key="rec_2"):
                st.info("This would generate a new song with the recommended parameters")
        
        with rec_cols[2]:
            st.markdown("#### Recommendation 3")
            st.markdown("**Style:** Acoustic")
            st.markdown("**Tempo:** 90-100 BPM")
            st.markdown("**Key:** G Major")
            st.markdown("**Voice:** More clarity")
            
            if st.button("Generate with these parameters", key="rec_3"):
                st.info("This would generate a new song with the recommended parameters")

# Footer
st.markdown("---")
st.markdown("© 2025 Song Creation System | Created with Streamlit")
