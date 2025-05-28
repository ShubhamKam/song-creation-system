# Song Creation System - Streamlit Cloud Deployment

This repository contains the Song Creation System implemented as a Streamlit application for permanent deployment on Streamlit Cloud.

## Features

- **YouTube Audio Extraction**: Download and process high-quality audio from any YouTube link
- **Audio Analysis**: Analyze rhythm, harmony, timbre, and structure of songs
- **Song Generation**: Create new compositions based on analyzed features with multiple AI models
- **Voice Customization**: Apply your voice characteristics to generated songs
- **Feedback System**: Provide feedback to improve future generations
- **Knowledge Base**: Build a personalized database of musical preferences

## Deployment on Streamlit Cloud

This repository is configured for direct deployment on Streamlit Cloud.

## File Structure

- `app_integrated.py`: Main Streamlit application (entry point)
- `youtube_extractor.py`: Module for extracting audio from YouTube
- `audio_analyzer.py`: Module for analyzing audio features
- `song_generator.py`: Module for generating songs
- `voice_customizer.py`: Module for customizing voice characteristics
- `knowledge_base.py`: Module for storing and retrieving song data and user feedback
- `requirements.txt`: List of Python dependencies

## Local Development

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the application: `streamlit run app_integrated.py`
4. Access at http://localhost:8501

## License

MIT License
