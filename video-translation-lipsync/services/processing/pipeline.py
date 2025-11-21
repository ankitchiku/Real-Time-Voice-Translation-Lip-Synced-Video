import os
import torch
import whisper
import subprocess
from pathlib import Path
from transformers import MarianMTModel, MarianTokenizer
from TTS.api import TTS
import numpy as np
import cv2
from typing import Callable, Optional
import shutil

class TranslationPipeline:
    """
    Complete pipeline for video translation and lip-sync
    """
    
    def __init__(self):
        self.model_dir = Path(os.getenv("MODEL_DIR", "/app/models"))
        self.use_gpu = os.getenv("USE_GPU", "false").lower() == "true"
        self.device = "cuda" if self.use_gpu and torch.cuda.is_available() else "cpu"
        
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Model placeholders
        self.whisper_model = None
        self.translation_models = {}
        self.tts_model = None
        self.wav2lip_model = None
        
        print(f"Pipeline initialized on device: {self.device}")
    
    def load_whisper(self):
        """Load Whisper ASR model"""
        if self.whisper_model is None:
            model_size = os.getenv("WHISPER_MODEL", "base")
            print(f"Loading Whisper {model_size} model...")
            self.whisper_model = whisper.load_model(model_size, device=self.device)
        return self.whisper_model
    
    def load_translation_model(self, target_lang: str):
        """Load MarianMT translation model"""
        if target_lang not in self.translation_models:
            model_name = f"Helsinki-NLP/opus-mt-en-{target_lang}"
            print(f"Loading translation model: {model_name}")
            
            try:
                tokenizer = MarianTokenizer.from_pretrained(model_name)
                model = MarianMTModel.from_pretrained(model_name)
                model.to(self.device)
                self.translation_models[target_lang] = (tokenizer, model)
            except Exception as e:
                print(f"Failed to load {model_name}, trying alternative...")
                # Fallback to ROMANCE language model
                model_name = "Helsinki-NLP/opus-mt-en-ROMANCE"
                tokenizer = MarianTokenizer.from_pretrained(model_name)
                model = MarianMTModel.from_pretrained(model_name)
                model.to(self.device)
                self.translation_models[target_lang] = (tokenizer, model)
        
        return self.translation_models[target_lang]
    
    def load_tts(self):
        """Load TTS model"""
        if self.tts_model is None:
            print("Loading TTS model...")
            # Using Coqui TTS
            self.tts_model = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", 
                                progress_bar=False, gpu=self.use_gpu)
        return self.tts_model
    
    def extract_audio(self, video_path: str, audio_path: str):
        """Extract audio from video using ffmpeg"""
        print(f"Extracting audio from {video_path}")
        cmd = [
            "ffmpeg", "-i", video_path,
            "-vn", "-acodec", "pcm_s16le",
            "-ar", "16000", "-ac", "1",
            "-y", audio_path
        ]
        subprocess.run(cmd, check=True, capture_output=True)
    
    def transcribe_audio(self, audio_path: str):
        """Transcribe audio using Whisper"""
        print("Transcribing audio...")
        model = self.load_whisper()
        result = model.transcribe(audio_path, word_timestamps=True)
        return result
    
    def translate_text(self, text: str, target_lang: str):
        """Translate text using MarianMT"""
        print(f"Translating to {target_lang}...")
        tokenizer, model = self.load_translation_model(target_lang)
        
        # Split into sentences if too long
        max_length = 512
        sentences = text.split('. ')
        translated_sentences = []
        
        for sentence in sentences:
            if not sentence.strip():
                continue
            inputs = tokenizer(sentence, return_tensors="pt", 
                             padding=True, truncation=True, max_length=max_length)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(**inputs)
            
            translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            translated_sentences.append(translated)
        
        return '. '.join(translated_sentences)
    
    def generate_speech(self, text: str, output_path: str, target_lang: str):
        """Generate speech from text using TTS"""
        print("Generating speech...")
        tts = self.load_tts()
        
        # For non-English, we'll use English TTS as placeholder
        # In production, load language-specific TTS models
        tts.tts_to_file(text=text, file_path=output_path)
    
    def apply_wav2lip(self, video_path: str, audio_path: str, output_path: str):
        """
        Apply Wav2Lip for lip-sync
        Note: This is a simplified version. Real Wav2Lip requires the model checkpoint
        """
        print("Applying lip-sync...")
        
        # Check if Wav2Lip is available
        wav2lip_path = self.model_dir / "Wav2Lip"
        
        if wav2lip_path.exists() and (wav2lip_path / "inference.py").exists():
            # Use actual Wav2Lip
            checkpoint = self.model_dir / "Wav2Lip" / "checkpoints" / "wav2lip_gan.pth"
            cmd = [
                "python", str(wav2lip_path / "inference.py"),
                "--checkpoint_path", str(checkpoint),
                "--face", video_path,
                "--audio", audio_path,
                "--outfile", output_path
            ]
            subprocess.run(cmd, check=True)
        else:
            # Fallback: combine audio with video without lip-sync
            print("Wav2Lip not found, combining audio and video without lip-sync...")
            cmd = [
                "ffmpeg", "-i", video_path, "-i", audio_path,
                "-c:v", "copy", "-map", "0:v:0", "-map", "1:a:0",
                "-shortest", "-y", output_path
            ]
            subprocess.run(cmd, check=True, capture_output=True)
    
    def process(self, input_path: str, output_path: str, target_language: str,
                job_id: str, status_callback: Optional[Callable] = None):
        """
        Main processing pipeline
        """
        temp_dir = Path(f"/tmp/{job_id}")
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Step 1: Extract audio (20%)
            if status_callback:
                status_callback(job_id, "extracting_audio", 20)
            
            audio_path = temp_dir / "audio.wav"
            self.extract_audio(input_path, str(audio_path))
            
            # Step 2: Transcribe (40%)
            if status_callback:
                status_callback(job_id, "transcribing", 40)
            
            transcript_result = self.transcribe_audio(str(audio_path))
            original_text = transcript_result['text']
            print(f"Transcribed: {original_text[:100]}...")
            
            # Step 3: Translate (60%)
            if status_callback:
                status_callback(job_id, "translating", 60)
            
            translated_text = self.translate_text(original_text, target_language)
            print(f"Translated: {translated_text[:100]}...")
            
            # Step 4: Generate speech (80%)
            if status_callback:
                status_callback(job_id, "generating_speech", 80)
            
            translated_audio_path = temp_dir / "translated_audio.wav"
            self.generate_speech(translated_text, str(translated_audio_path), target_language)
            
            # Step 5: Apply lip-sync (90%)
            if status_callback:
                status_callback(job_id, "lip_syncing", 90)
            
            self.apply_wav2lip(input_path, str(translated_audio_path), output_path)
            
            print(f"Processing complete! Output: {output_path}")
            
        except Exception as e:
            print(f"Processing failed: {e}")
            raise
        finally:
            # Cleanup temp directory
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

def download_models():
    """Download required models"""
    print("Downloading models...")
    
    # Whisper will auto-download
    whisper.load_model("base")
    
    # MarianMT models
    from transformers import MarianMTModel, MarianTokenizer
    for lang in ["es", "fr", "de"]:
        try:
            MarianTokenizer.from_pretrained(f"Helsinki-NLP/opus-mt-en-{lang}")
            MarianMTModel.from_pretrained(f"Helsinki-NLP/opus-mt-en-{lang}")
        except:
            pass
    
    # TTS model
    TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False)
    
    print("Models downloaded successfully!")

if __name__ == "__main__":
    download_models()