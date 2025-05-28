"""
Knowledge Base Module for Streamlit Song Creation System

This module handles the storage and retrieval of song analysis data and user feedback
for improving future song recommendations and generations.
"""

import os
import json
import logging
import time
import numpy as np
from typing import Dict, Any, List, Optional
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KnowledgeBase:
    """Class for managing the song creation knowledge base and recommendations."""
    
    def __init__(self, data_dir: str = None):
        """
        Initialize the knowledge base.
        
        Args:
            data_dir: Directory to store knowledge base data
        """
        self.data_dir = data_dir or os.path.join(os.path.expanduser("~"), ".song_creation_system")
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialize database files
        self.songs_db_file = os.path.join(self.data_dir, "songs_database.json")
        self.feedback_db_file = os.path.join(self.data_dir, "feedback_database.json")
        self.preferences_file = os.path.join(self.data_dir, "user_preferences.json")
        
        # Initialize databases
        self._initialize_databases()
        
        logger.info(f"Knowledge base initialized with data directory: {self.data_dir}")
    
    def _initialize_databases(self):
        """Initialize database files if they don't exist."""
        # Songs database
        if not os.path.exists(self.songs_db_file):
            with open(self.songs_db_file, 'w') as f:
                json.dump([], f)
        
        # Feedback database
        if not os.path.exists(self.feedback_db_file):
            with open(self.feedback_db_file, 'w') as f:
                json.dump([], f)
        
        # User preferences
        if not os.path.exists(self.preferences_file):
            with open(self.preferences_file, 'w') as f:
                json.dump({
                    "preferred_genres": [],
                    "preferred_tempo_range": [80, 160],
                    "preferred_key": None,
                    "preferred_complexity": 0.5,
                    "last_updated": time.strftime('%Y-%m-%d %H:%M:%S')
                }, f, indent=2)
    
    def add_analyzed_song(self, youtube_url: str, analysis_results: Dict[str, Any]) -> str:
        """
        Add an analyzed song to the knowledge base.
        
        Args:
            youtube_url: URL of the YouTube video
            analysis_results: Dictionary containing audio analysis results
            
        Returns:
            ID of the added song entry
        """
        logger.info(f"Adding analyzed song to knowledge base: {youtube_url}")
        
        try:
            # Load existing database
            with open(self.songs_db_file, 'r') as f:
                songs_db = json.load(f)
            
            # Generate a unique ID
            song_id = f"song_{int(time.time())}_{len(songs_db)}"
            
            # Extract key features for storage
            song_entry = {
                "id": song_id,
                "youtube_url": youtube_url,
                "date_added": time.strftime('%Y-%m-%d %H:%M:%S'),
                "features": {
                    "tempo": analysis_results.get('rhythm', {}).get('tempo', 120),
                    "key": analysis_results.get('harmony', {}).get('key', 'C'),
                    "mode": analysis_results.get('harmony', {}).get('mode', 'Major'),
                    "chord_complexity": analysis_results.get('harmony', {}).get('chord_complexity', 0.5),
                    "rhythm_regularity": analysis_results.get('rhythm', {}).get('rhythm_regularity', 0.5),
                    "instruments": analysis_results.get('timbre', {}).get('instruments', {}),
                    "sections": len(analysis_results.get('structure', {}).get('sections', [])),
                }
            }
            
            # Add to database
            songs_db.append(song_entry)
            
            # Save updated database
            with open(self.songs_db_file, 'w') as f:
                json.dump(songs_db, f, indent=2)
            
            logger.info(f"Song added to knowledge base with ID: {song_id}")
            return song_id
            
        except Exception as e:
            logger.error(f"Error adding song to knowledge base: {str(e)}")
            raise
    
    def add_generation_feedback(self, song_id: str, generation_info: Dict[str, Any], 
                              rating: int, feedback_text: str) -> str:
        """
        Add user feedback for a generated song.
        
        Args:
            song_id: ID of the original analyzed song
            generation_info: Dictionary containing generation parameters and results
            rating: User rating (1-5)
            feedback_text: User feedback text
            
        Returns:
            ID of the feedback entry
        """
        logger.info(f"Adding generation feedback for song: {song_id}")
        
        try:
            # Load existing database
            with open(self.feedback_db_file, 'r') as f:
                feedback_db = json.load(f)
            
            # Generate a unique ID
            feedback_id = f"feedback_{int(time.time())}_{len(feedback_db)}"
            
            # Create feedback entry
            feedback_entry = {
                "id": feedback_id,
                "song_id": song_id,
                "date_added": time.strftime('%Y-%m-%d %H:%M:%S'),
                "rating": rating,
                "feedback_text": feedback_text,
                "generation_params": {
                    "model": generation_info.get('model', 'unknown'),
                    "creativity": generation_info.get('creativity', 0.7),
                    "similarity": generation_info.get('similarity', 0.5),
                    "tempo": generation_info.get('tempo', 120),
                    "key": generation_info.get('key', 'C'),
                    "mode": generation_info.get('mode', 'Major')
                }
            }
            
            # Add to database
            feedback_db.append(feedback_entry)
            
            # Save updated database
            with open(self.feedback_db_file, 'w') as f:
                json.dump(feedback_db, f, indent=2)
            
            # Update user preferences based on feedback
            self._update_preferences_from_feedback(feedback_entry)
            
            logger.info(f"Feedback added with ID: {feedback_id}")
            return feedback_id
            
        except Exception as e:
            logger.error(f"Error adding feedback: {str(e)}")
            raise
    
    def _update_preferences_from_feedback(self, feedback_entry: Dict[str, Any]):
        """Update user preferences based on feedback."""
        logger.info("Updating user preferences based on feedback")
        
        try:
            # Load current preferences
            with open(self.preferences_file, 'r') as f:
                preferences = json.load(f)
            
            # Only update preferences for positive ratings (4-5)
            if feedback_entry['rating'] >= 4:
                # Get generation parameters
                params = feedback_entry['generation_params']
                
                # Update tempo preference (weighted average)
                current_min, current_max = preferences['preferred_tempo_range']
                new_tempo = params.get('tempo', 120)
                
                # Adjust range to include the new tempo
                if new_tempo < current_min:
                    preferences['preferred_tempo_range'][0] = max(60, new_tempo - 10)
                if new_tempo > current_max:
                    preferences['preferred_tempo_range'][1] = min(200, new_tempo + 10)
                
                # Update key preference (if consistently preferred)
                new_key = params.get('key', None)
                if new_key:
                    # Load all feedback to check for key preference patterns
                    with open(self.feedback_db_file, 'r') as f:
                        all_feedback = json.load(f)
                    
                    # Count positive ratings for each key
                    key_ratings = {}
                    for fb in all_feedback:
                        if fb['rating'] >= 4:
                            key = fb['generation_params'].get('key', None)
                            if key:
                                key_ratings[key] = key_ratings.get(key, 0) + 1
                    
                    # If this key has multiple positive ratings, set as preference
                    if key_ratings.get(new_key, 0) >= 2:
                        preferences['preferred_key'] = new_key
                
                # Update complexity preference (weighted average)
                current_complexity = preferences['preferred_complexity']
                new_complexity = params.get('creativity', 0.7)
                preferences['preferred_complexity'] = (current_complexity * 0.7) + (new_complexity * 0.3)
            
            # Update timestamp
            preferences['last_updated'] = time.strftime('%Y-%m-%d %H:%M:%S')
            
            # Save updated preferences
            with open(self.preferences_file, 'w') as f:
                json.dump(preferences, f, indent=2)
            
            logger.info("User preferences updated")
            
        except Exception as e:
            logger.error(f"Error updating preferences: {str(e)}")
    
    def get_recommendations(self, num_recommendations: int = 3) -> List[Dict[str, Any]]:
        """
        Get song generation recommendations based on user history and preferences.
        
        Args:
            num_recommendations: Number of recommendations to generate
            
        Returns:
            List of recommendation dictionaries
        """
        logger.info(f"Generating {num_recommendations} recommendations")
        
        try:
            # Load user preferences
            with open(self.preferences_file, 'r') as f:
                preferences = json.load(f)
            
            # Load feedback history
            with open(self.feedback_db_file, 'r') as f:
                feedback_db = json.load(f)
            
            # Generate recommendations
            recommendations = []
            
            # Base recommendation on preferences
            base_recommendation = {
                "style": "Pop/Rock",  # Default style
                "tempo_range": preferences['preferred_tempo_range'],
                "key": preferences['preferred_key'] or "C Major",
                "complexity": preferences['preferred_complexity'],
                "model": "neural_network",
                "description": "Based on your overall preferences"
            }
            recommendations.append(base_recommendation)
            
            # If we have feedback history, generate more targeted recommendations
            if feedback_db:
                # Find highest rated generations
                sorted_feedback = sorted(feedback_db, key=lambda x: x['rating'], reverse=True)
                
                if len(sorted_feedback) > 0 and len(recommendations) < num_recommendations:
                    # Recommendation based on highest rated song
                    top_rated = sorted_feedback[0]
                    params = top_rated['generation_params']
                    
                    # Slightly modify parameters for variety
                    rec = {
                        "style": "Similar to your highest rated song",
                        "tempo_range": [
                            max(60, params.get('tempo', 120) - 10),
                            min(200, params.get('tempo', 120) + 10)
                        ],
                        "key": params.get('key', 'C') + " " + params.get('mode', 'Major'),
                        "complexity": min(1.0, params.get('creativity', 0.7) + 0.1),
                        "model": params.get('model', 'neural_network'),
                        "description": "Based on your highest rated song"
                    }
                    recommendations.append(rec)
                
                if len(sorted_feedback) > 2 and len(recommendations) < num_recommendations:
                    # Find most common model among high-rated songs
                    high_rated = [fb for fb in sorted_feedback if fb['rating'] >= 4]
                    
                    if high_rated:
                        # Count model occurrences
                        model_counts = {}
                        for fb in high_rated:
                            model = fb['generation_params'].get('model', 'neural_network')
                            model_counts[model] = model_counts.get(model, 0) + 1
                        
                        # Get most common model
                        most_common_model = max(model_counts.items(), key=lambda x: x[1])[0]
                        
                        # Create recommendation with this model but different parameters
                        rec = {
                            "style": f"Using your preferred {most_common_model} model",
                            "tempo_range": [
                                preferences['preferred_tempo_range'][0] + 10,
                                preferences['preferred_tempo_range'][1] - 10
                            ],
                            "key": "G Major" if preferences['preferred_key'] != "G" else "D Major",
                            "complexity": max(0.3, preferences['preferred_complexity'] - 0.2),
                            "model": most_common_model,
                            "description": f"Exploring new sounds with your preferred {most_common_model} model"
                        }
                        recommendations.append(rec)
            
            # Fill remaining recommendations with creative variations
            while len(recommendations) < num_recommendations:
                # Generate a creative variation
                styles = ["Electronic", "Acoustic", "Jazz-inspired", "Classical fusion"]
                models = ["neural_network", "markov_chain", "transformer", "hybrid"]
                
                rec = {
                    "style": styles[len(recommendations) % len(styles)],
                    "tempo_range": [
                        max(60, np.random.randint(80, 140)),
                        min(200, np.random.randint(100, 160))
                    ],
                    "key": ["C", "G", "D", "A", "F"][np.random.randint(0, 5)] + " " + 
                           ["Major", "Minor"][np.random.randint(0, 2)],
                    "complexity": np.random.uniform(0.4, 0.8),
                    "model": models[len(recommendations) % len(models)],
                    "description": "Exploring new musical territory"
                }
                recommendations.append(rec)
            
            logger.info(f"Generated {len(recommendations)} recommendations")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return []
    
    def get_song_history(self) -> List[Dict[str, Any]]:
        """
        Get history of analyzed songs and their generations.
        
        Returns:
            List of song history entries with feedback
        """
        logger.info("Retrieving song history")
        
        try:
            # Load databases
            with open(self.songs_db_file, 'r') as f:
                songs_db = json.load(f)
            
            with open(self.feedback_db_file, 'r') as f:
                feedback_db = json.load(f)
            
            # Create a lookup for feedback by song_id
            feedback_lookup = {}
            for feedback in feedback_db:
                song_id = feedback['song_id']
                if song_id not in feedback_lookup:
                    feedback_lookup[song_id] = []
                feedback_lookup[song_id].append(feedback)
            
            # Combine song data with feedback
            history = []
            for song in songs_db:
                song_id = song['id']
                entry = {
                    "id": song_id,
                    "youtube_url": song['youtube_url'],
                    "date_added": song['date_added'],
                    "features": song['features'],
                    "feedback": feedback_lookup.get(song_id, [])
                }
                history.append(entry)
            
            # Sort by date (newest first)
            history.sort(key=lambda x: x['date_added'], reverse=True)
            
            logger.info(f"Retrieved history with {len(history)} entries")
            return history
            
        except Exception as e:
            logger.error(f"Error retrieving history: {str(e)}")
            return []
    
    def analyze_feedback_with_llm(self, feedback_text: str) -> Dict[str, Any]:
        """
        Analyze user feedback using LLM to extract actionable insights.
        
        Args:
            feedback_text: User feedback text
            
        Returns:
            Dictionary containing analysis results
        """
        logger.info("Analyzing feedback with LLM")
        
        # In a real implementation, this would use an actual LLM API
        # For this example, we'll simulate the analysis
        
        # Simple keyword-based analysis
        keywords = {
            "tempo": ["tempo", "speed", "fast", "slow", "bpm", "pace"],
            "rhythm": ["rhythm", "beat", "groove", "rhythmic", "timing"],
            "melody": ["melody", "tune", "melodic", "catchy", "hook"],
            "harmony": ["harmony", "chord", "harmonic", "key", "tone"],
            "structure": ["structure", "section", "verse", "chorus", "bridge", "arrangement"],
            "complexity": ["complex", "complexity", "simple", "complicated", "intricate"],
            "voice": ["voice", "vocal", "sing", "singing", "vocalist"]
        }
        
        # Count keyword occurrences
        counts = {category: 0 for category in keywords}
        for category, words in keywords.items():
            for word in words:
                counts[category] += feedback_text.lower().count(word)
        
        # Determine sentiment for each category
        sentiment = {category: 0 for category in keywords}  # -1 negative, 0 neutral, 1 positive
        
        positive_words = ["good", "great", "excellent", "like", "love", "amazing", "awesome", "fantastic", "better"]
        negative_words = ["bad", "poor", "terrible", "dislike", "hate", "awful", "worse", "disappointing"]
        
        for category in keywords:
            # Check for positive sentiment
            for word in positive_words:
                # Look for positive words near category keywords
                for keyword in keywords[category]:
                    if keyword in feedback_text.lower() and word in feedback_text.lower():
                        # Very simple proximity check
                        if abs(feedback_text.lower().find(keyword) - feedback_text.lower().find(word)) < 20:
                            sentiment[category] += 1
            
            # Check for negative sentiment
            for word in negative_words:
                # Look for negative words near category keywords
                for keyword in keywords[category]:
                    if keyword in feedback_text.lower() and word in feedback_text.lower():
                        # Very simple proximity check
                        if abs(feedback_text.lower().find(keyword) - feedback_text.lower().find(word)) < 20:
                            sentiment[category] -= 1
        
        # Generate suggestions based on analysis
        suggestions = []
        
        if sentiment["tempo"] < 0:
            if "fast" in feedback_text.lower() or "too quick" in feedback_text.lower():
                suggestions.append("Reduce tempo in future generations")
            elif "slow" in feedback_text.lower() or "too slow" in feedback_text.lower():
                suggestions.append("Increase tempo in future generations")
            else:
                suggestions.append("Adjust tempo based on user preference")
        
        if sentiment["rhythm"] < 0:
            if counts["rhythm"] > 0:
                if "complex" in feedback_text.lower():
                    suggestions.append("Simplify rhythm patterns")
                elif "simple" in feedback_text.lower() or "boring" in feedback_text.lower():
                    suggestions.append("Add more rhythmic variation")
                else:
                    suggestions.append("Experiment with different rhythm patterns")
        
        if sentiment["melody"] < 0:
            suggestions.append("Focus on more memorable melodic patterns")
        
        if sentiment["harmony"] < 0:
            if "complex" in feedback_text.lower():
                suggestions.append("Use simpler chord progressions")
            else:
                suggestions.append("Experiment with different chord progressions")
        
        if sentiment["voice"] < 0:
            if "pitch" in feedback_text.lower():
                suggestions.append("Adjust voice pitch settings")
            elif "clarity" in feedback_text.lower():
                suggestions.append("Increase voice clarity parameter")
            else:
                suggestions.append("Fine-tune voice customization parameters")
        
        # Determine overall sentiment
        overall_sentiment = sum(sentiment.values())
        if overall_sentiment > 2:
            overall = "positive"
        elif overall_sentiment < -2:
            overall = "negative"
        else:
            overall = "neutral"
        
        # Generate parameter adjustments
        parameter_adjustments = {}
        
        if sentiment["tempo"] < 0:
            if "fast" in feedback_text.lower():
                parameter_adjustments["tempo_adjustment"] = -10
            elif "slow" in feedback_text.lower():
                parameter_adjustments["tempo_adjustment"] = 10
        
        if sentiment["complexity"] < 0:
            if "complex" in feedback_text.lower() or "complicated" in feedback_text.lower():
                parameter_adjustments["complexity_adjustment"] = -0.2
            elif "simple" in feedback_text.lower() or "boring" in feedback_text.lower():
                parameter_adjustments["complexity_adjustment"] = 0.2
        
        if sentiment["voice"] < 0:
            if "pitch" in feedback_text.lower() and "high" in feedback_text.lower():
                parameter_adjustments["voice_pitch_adjustment"] = -2
            elif "pitch" in feedback_text.lower() and "low" in feedback_text.lower():
                parameter_adjustments["voice_pitch_adjustment"] = 2
        
        analysis_result = {
            "keyword_counts": counts,
            "sentiment_analysis": sentiment,
            "overall_sentiment": overall,
            "suggestions": suggestions,
            "parameter_adjustments": parameter_adjustments
        }
        
        logger.info(f"Feedback analysis completed with {len(suggestions)} suggestions")
        return analysis_result
