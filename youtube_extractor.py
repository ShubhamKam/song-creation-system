"""
YouTube Audio Extractor Module for Streamlit Song Creation System

This module handles the extraction of audio from YouTube videos using yt-dlp.
"""

import os
import yt_dlp
import tempfile
import logging
from typing import Dict, Any, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YouTubeExtractor:
    """Class for extracting audio from YouTube videos."""
    
    def __init__(self, output_dir: str = None):
        """
        Initialize the YouTube extractor.
        
        Args:
            output_dir: Directory to save extracted audio files
        """
        self.output_dir = output_dir or tempfile.gettempdir()
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"YouTube extractor initialized with output directory: {self.output_dir}")
    
    def extract_audio(self, youtube_url: str, quality: str = "high", 
                     format: str = "mp3", progress_callback=None) -> Tuple[str, Dict[str, Any]]:
        """
        Extract audio from a YouTube video.
        
        Args:
            youtube_url: URL of the YouTube video
            quality: Audio quality (low, medium, high)
            format: Audio format (mp3, wav, flac)
            progress_callback: Optional callback function for progress updates
            
        Returns:
            Tuple containing:
                - Path to the extracted audio file
                - Dictionary with video metadata
        """
        logger.info(f"Extracting audio from: {youtube_url}")
        
        # Map quality to yt-dlp audio quality
        quality_map = {
            "low": "128",
            "medium": "192",
            "high": "320"
        }
        audio_quality = quality_map.get(quality.lower(), "320")
        
        # Prepare output filename template
        output_template = os.path.join(self.output_dir, "%(title)s.%(ext)s")
        
        # Configure yt-dlp options
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': format.lower(),
                'preferredquality': audio_quality,
            }],
            'outtmpl': output_template,
            'noplaylist': True,
            'quiet': True,
            'no_warnings': True,
        }
        
        # Add progress hook if callback provided
        if progress_callback:
            def progress_hook(d):
                if d['status'] == 'downloading':
                    if 'total_bytes' in d and d['total_bytes'] > 0:
                        progress = d['downloaded_bytes'] / d['total_bytes']
                    elif 'total_bytes_estimate' in d and d['total_bytes_estimate'] > 0:
                        progress = d['downloaded_bytes'] / d['total_bytes_estimate']
                    else:
                        progress = 0
                    progress_callback(min(progress, 1.0))
                elif d['status'] == 'finished':
                    progress_callback(1.0)
            
            ydl_opts['progress_hooks'] = [progress_hook]
        
        # Extract audio
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=True)
                if 'entries' in info:
                    # Playlist, take first video
                    info = info['entries'][0]
                
                # Get the output filename
                output_file = ydl.prepare_filename(info)
                base, _ = os.path.splitext(output_file)
                output_file = f"{base}.{format.lower()}"
                
                # Ensure file exists
                if not os.path.exists(output_file):
                    raise FileNotFoundError(f"Expected output file not found: {output_file}")
                
                # Prepare metadata
                metadata = {
                    'title': info.get('title', 'Unknown'),
                    'uploader': info.get('uploader', 'Unknown'),
                    'duration': info.get('duration', 0),
                    'upload_date': info.get('upload_date', 'Unknown'),
                    'view_count': info.get('view_count', 0),
                    'thumbnail': info.get('thumbnail', None),
                    'description': info.get('description', ''),
                    'categories': info.get('categories', []),
                    'tags': info.get('tags', []),
                }
                
                logger.info(f"Successfully extracted audio to: {output_file}")
                return output_file, metadata
                
        except Exception as e:
            logger.error(f"Error extracting audio: {str(e)}")
            raise
    
    def get_video_info(self, youtube_url: str) -> Dict[str, Any]:
        """
        Get information about a YouTube video without downloading.
        
        Args:
            youtube_url: URL of the YouTube video
            
        Returns:
            Dictionary with video metadata
        """
        logger.info(f"Getting info for: {youtube_url}")
        
        ydl_opts = {
            'format': 'bestaudio/best',
            'noplaylist': True,
            'quiet': True,
            'no_warnings': True,
            'skip_download': True,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=False)
                if 'entries' in info:
                    # Playlist, take first video
                    info = info['entries'][0]
                
                # Prepare metadata
                metadata = {
                    'title': info.get('title', 'Unknown'),
                    'uploader': info.get('uploader', 'Unknown'),
                    'duration': info.get('duration', 0),
                    'upload_date': info.get('upload_date', 'Unknown'),
                    'view_count': info.get('view_count', 0),
                    'thumbnail': info.get('thumbnail', None),
                    'description': info.get('description', ''),
                    'categories': info.get('categories', []),
                    'tags': info.get('tags', []),
                }
                
                logger.info(f"Successfully retrieved info for: {youtube_url}")
                return metadata
                
        except Exception as e:
            logger.error(f"Error getting video info: {str(e)}")
            raise
    
    def validate_youtube_url(self, url: str) -> bool:
        """
        Validate if a URL is a valid YouTube URL.
        
        Args:
            url: URL to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not url:
            return False
        
        # Simple validation
        valid_prefixes = [
            'https://www.youtube.com/watch?v=',
            'https://youtube.com/watch?v=',
            'https://youtu.be/',
            'https://www.youtube.com/shorts/',
            'https://youtube.com/shorts/'
        ]
        
        return any(url.startswith(prefix) for prefix in valid_prefixes)
