"""
Emotion-Based Music Player
Uses webcam to detect facial emotions and plays music accordingly.
"""

import cv2
import pygame
import os
import random
import time
import numpy as np
from collections import deque
from fer import FER

class EmotionMusicPlayer:
    def __init__(self, songs_dir="songs"):
        """
        Initialize the Emotion Music Player
        
        Args:
            songs_dir: Directory containing emotion-based song folders
        """
        self.songs_dir = songs_dir
        
        # Initialize emotion detector (using default for stability)
        # MTCNN can be enabled for better accuracy but may cause issues on some systems
        use_mtcnn = False  # Set to True to try MTCNN (may crash on some systems)
        
        if use_mtcnn:
            try:
                self.emotion_detector = FER(mtcnn=True)
                print("✓ Using MTCNN for enhanced face detection accuracy")
            except Exception as e:
                print(f"⚠ MTCNN initialization failed, using default detector: {e}")
                self.emotion_detector = FER(mtcnn=False)
                print("✓ Using default Haar Cascade detector")
        else:
            self.emotion_detector = FER(mtcnn=False)
            print("✓ Using default Haar Cascade detector (stable)")
        
        # Initialize pygame mixer for audio playback
        pygame.mixer.init()
        
        # Track current state
        self.current_emotion = None
        self.current_song = None
        self.last_emotion_change = time.time()
        self.emotion_stability_threshold = 3  # seconds to wait before changing music
        
        # Emotion smoothing parameters for improved accuracy
        self.emotion_history = deque(maxlen=10)  # Store last 10 emotion readings
        self.confidence_threshold = 0.3  # Minimum confidence for emotion detection
        self.detection_frequency = 3  # Analyze every 3 frames (increased from 5)
        
        # Emotion mapping (fer emotions to our simplified categories)
        self.emotion_categories = {
            'happy': ['happy'],
            'sad': ['sad', 'fear', 'angry'],
            'neutral': ['neutral', 'surprise', 'disgust']
        }
        
        # Load available songs
        self.songs = self.load_songs()
        
        print("Emotion Music Player initialized!")
        print(f"Songs loaded: {sum(len(v) for v in self.songs.values())} total")
        
    def load_songs(self):
        """Load songs from emotion-based folders"""
        songs = {'happy': [], 'sad': [], 'neutral': []}
        
        for emotion in ['happy', 'sad', 'neutral']:
            folder_path = os.path.join(self.songs_dir, emotion)
            
            if os.path.exists(folder_path):
                for file in os.listdir(folder_path):
                    if file.endswith(('.mp3', '.wav', '.ogg', '.flac')):
                        songs[emotion].append(os.path.join(folder_path, file))
                        
            if not songs[emotion]:
                print(f"Warning: No songs found in {folder_path}")
                
        return songs
    
    def get_dominant_emotion_category(self, emotions, return_confidence=False):
        """
        Convert fer emotion dictionary to our simplified categories
        
        Args:
            emotions: Dictionary of emotions from fer
            return_confidence: If True, also return the confidence score
            
        Returns:
            'happy', 'sad', or 'neutral' (and confidence if requested)
        """
        if not emotions:
            return (None, 0.0) if return_confidence else None
            
        # Get the emotion with highest score
        dominant_emotion, confidence = max(emotions.items(), key=lambda x: x[1])
        
        # Check if confidence meets threshold
        if confidence < self.confidence_threshold:
            return (None, confidence) if return_confidence else None
        
        # Map to our categories
        category = 'neutral'  # default
        for cat, emotion_list in self.emotion_categories.items():
            if dominant_emotion in emotion_list:
                category = cat
                break
        
        if return_confidence:
            return category, confidence
        return category
    
    def get_smoothed_emotion(self):
        """
        Get smoothed emotion based on recent history to reduce jitter
        
        Returns:
            Most common emotion from recent history, or None
        """
        if len(self.emotion_history) < 3:
            return None
        
        # Count occurrences of each emotion
        emotion_counts = {}
        for emotion in self.emotion_history:
            if emotion:
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        if not emotion_counts:
            return None
        
        # Return the most common emotion
        return max(emotion_counts.items(), key=lambda x: x[1])[0]
    
    def preprocess_frame(self, frame):
        """
        Preprocess frame for better face detection
        
        Args:
            frame: Input frame
            
        Returns:
            Preprocessed frame
        """
        # Enhance contrast using CLAHE
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def play_song_for_emotion(self, emotion):
        """
        Play a random song from the emotion's folder
        
        Args:
            emotion: 'happy', 'sad', or 'neutral'
        """
        if emotion not in self.songs or not self.songs[emotion]:
            print(f"No songs available for emotion: {emotion}")
            return
            
        # Select a random song from the emotion category
        song = random.choice(self.songs[emotion])
        
        # Stop current song if playing
        pygame.mixer.music.stop()
        
        # Load and play new song
        try:
            pygame.mixer.music.load(song)
            pygame.mixer.music.play()
            self.current_song = song
            print(f"\n🎵 Playing ({emotion}): {os.path.basename(song)}")
        except Exception as e:
            print(f"Error playing song: {e}")
    
    def run(self):
        """Main loop for emotion detection and music playback"""
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not access webcam")
            return
            
        print("\n" + "="*60)
        print("Emotion Music Player Running!")
        print("="*60)
        print("Instructions:")
        print("- Look at the webcam")
        print("- Your emotion will be detected in real-time")
        print("- Music will play based on your emotion")
        print("- Press 'q' to quit")
        print("="*60 + "\n")
        
        frame_count = 0
        pending_emotion = None
        pending_emotion_time = None
        last_detected_emotion = None
        last_confidence = 0.0
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Could not read frame")
                break
                
            # Mirror the frame for better user experience
            frame = cv2.flip(frame, 1)
            
            # Detect emotions every N frames (configurable for performance)
            if frame_count % self.detection_frequency == 0:
                # Preprocess frame for better detection
                processed_frame = self.preprocess_frame(frame)
                
                # Detect emotions in the frame
                try:
                    result = self.emotion_detector.detect_emotions(processed_frame)
                except Exception as e:
                    print(f"Detection error: {e}")
                    result = None
                
                if result and len(result) > 0:
                    # Get the largest face (most likely to be the user)
                    face = max(result, key=lambda x: x['box'][2] * x['box'][3])
                    emotions = face['emotions']
                    box = face['box']
                    
                    # Get emotion category with confidence
                    emotion_category, confidence = self.get_dominant_emotion_category(emotions, return_confidence=True)
                    
                    if emotion_category:
                        # Add to history for smoothing
                        self.emotion_history.append(emotion_category)
                        last_detected_emotion = emotion_category
                        last_confidence = confidence
                        
                        # Get smoothed emotion
                        smoothed_emotion = self.get_smoothed_emotion()
                        
                        # Draw bounding box around face
                        x, y, w, h = box
                        
                        # Color-code box based on confidence
                        if confidence > 0.7:
                            box_color = (0, 255, 0)  # Green - high confidence
                        elif confidence > 0.5:
                            box_color = (0, 255, 255)  # Yellow - medium confidence
                        else:
                            box_color = (0, 165, 255)  # Orange - low confidence
                        
                        cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, 2)
                        
                        # Display emotion and confidence
                        cv2.putText(frame, f"{emotion_category} ({confidence:.0%})", 
                                  (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.9, box_color, 2)
                        
                        # Use smoothed emotion for music selection
                        if smoothed_emotion and smoothed_emotion != self.current_emotion:
                            if pending_emotion == smoothed_emotion:
                                # Same pending emotion, check if enough time has passed
                                if time.time() - pending_emotion_time >= self.emotion_stability_threshold:
                                    # Emotion is stable, change music
                                    self.current_emotion = smoothed_emotion
                                    self.play_song_for_emotion(smoothed_emotion)
                                    pending_emotion = None
                            else:
                                # New emotion detected, start timer
                                pending_emotion = smoothed_emotion
                                pending_emotion_time = time.time()
                        elif smoothed_emotion == self.current_emotion:
                            # Same emotion as current, reset pending
                            pending_emotion = None
                            
                            # Check if current song has finished
                            if not pygame.mixer.music.get_busy() and self.current_emotion:
                                self.play_song_for_emotion(self.current_emotion)
            
            # Display current emotion and song info on frame
            info_y = 30
            
            # Show detection status
            detection_status = f"Confidence: {last_confidence:.0%}" if last_detected_emotion else "Detecting..."
            cv2.putText(frame, detection_status, 
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            
            # Show current locked-in emotion
            cv2.putText(frame, f"Current Emotion: {self.current_emotion or 'None'}", 
                       (10, info_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show emotion history for debugging
            if len(self.emotion_history) > 0:
                history_str = f"History: {list(self.emotion_history)[-5:]}"  # Last 5
                cv2.putText(frame, history_str[:50], 
                           (10, info_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            
            if self.current_song:
                song_name = os.path.basename(self.current_song)
                cv2.putText(frame, f"Playing: {song_name}", 
                           (10, info_y + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            if pending_emotion:
                remaining = self.emotion_stability_threshold - (time.time() - pending_emotion_time)
                if remaining > 0:
                    cv2.putText(frame, f"Switching to {pending_emotion} in {remaining:.1f}s", 
                               (10, info_y + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Display the frame
            cv2.imshow('Emotion Music Player - Press Q to quit', frame)
            
            # Check for quit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nStopping Emotion Music Player...")
                break
                
            frame_count += 1
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        pygame.mixer.music.stop()
        pygame.mixer.quit()
        print("Emotion Music Player stopped.")


def main():
    """Main entry point"""
    player = EmotionMusicPlayer(songs_dir="songs")
    player.run()


if __name__ == "__main__":
    main()

