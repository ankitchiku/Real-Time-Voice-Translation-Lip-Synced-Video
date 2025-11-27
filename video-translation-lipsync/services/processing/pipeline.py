import os
import torch
import whisper
import subprocess
from pathlib import Path
from transformers import MarianMTModel, MarianTokenizer, AutoTokenizer, AutoModelForSeq2SeqLM
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
            # Primary: Helsinki models (en -> target)
            model_name = f"Helsinki-NLP/opus-mt-en-{target_lang}"
            print(f"Loading translation model: {model_name}")

            try:
                tokenizer = MarianTokenizer.from_pretrained(model_name)
                model = MarianMTModel.from_pretrained(model_name)
                model.to(self.device)
                self.translation_models[target_lang] = (tokenizer, model)
            except Exception:
                # Fallback: try a multilingual seq2seq (m2m100) which supports many languages
                print(f"Falling back to m2m100 model for language: {target_lang}")
                m2m_name = "facebook/m2m100_418M"
                try:
                    tokenizer = AutoTokenizer.from_pretrained(m2m_name)
                    model = AutoModelForSeq2SeqLM.from_pretrained(m2m_name)
                    model.to(self.device)
                    self.translation_models[target_lang] = (tokenizer, model)
                except Exception as e:
                    print(f"Failed to load fallback model {m2m_name}: {e}")
                    raise

        return self.translation_models[target_lang]
    
    def load_tts(self):
        """Load TTS model"""
        if self.tts_model is None:
            print("Loading TTS model...")
            # Using Coqui TTS
            self.tts_model = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", 
                                progress_bar=False, gpu=self.use_gpu)
        return self.tts_model

    def compute_speaker_embedding(self, speaker_wav_path: str):
        """Attempt to compute a speaker embedding from a reference WAV file.

        This is a lightweight helper: if the installed TTS model exposes a
        `compute_embedding` or similar API, it will be used. Otherwise returns None.
        """
        if not speaker_wav_path or not Path(speaker_wav_path).exists():
            return None

        tts = self.load_tts()
        # Best-effort: Coqui TTS multi-speaker models sometimes expose `tts.speaker_manager` or
        # `tts.compute_embedding` APIs. We attempt common ones, but don't fail if missing.
        try:
            if hasattr(tts, "compute_embedding"):
                return tts.compute_embedding(speaker_wav_path)
            if hasattr(tts, "speaker_manager") and hasattr(tts.speaker_manager, "create_embedding"):
                return tts.speaker_manager.create_embedding(speaker_wav_path)
        except Exception as e:
            print(f"Speaker embedding extraction not available: {e}")

        return None
    
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

        # We must chunk by tokenizer tokens (<=512) to avoid silent truncation which changes content.
        max_length = 512
        # Use the tokenizer to split text into token-aware chunks.
        tokens = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
        chunks = []
        current = []
        for t in tokens.tolist():
            current.append(t)
            if len(current) >= max_length - 2:  # leave room for special tokens
                chunks.append(current)
                current = []
        if current:
            chunks.append(current)

        translated_pieces = []
        for chunk in chunks:
            inputs = {"input_ids": torch.tensor([chunk]).to(self.device)}
            with torch.no_grad():
                outputs = model.generate(**inputs, max_length=max_length)
            translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            translated_pieces.append(translated)

        # Join with spacing, try to preserve sentence boundaries if possible
        return ' '.join(p.strip() for p in translated_pieces if p.strip())
    
    def generate_speech(self, text: str, output_path: str, target_lang: str):
        """Generate speech from text using TTS"""
        print("Generating speech...")
        tts = self.load_tts()
        # Optionally allow a speaker reference file to try to clone voice.
        # If MODEL_SPEAKER_REF is set to a path, and the TTS implementation supports speaker embeddings,
        # attempt to use it. Otherwise fallback to the base model.
        speaker_ref = os.getenv("MODEL_SPEAKER_REF", None)

        # Try to compute an embedding if a reference is provided and feed it to TTS where supported.
        embedding = None
        if speaker_ref and Path(speaker_ref).exists():
            embedding = self.compute_speaker_embedding(speaker_ref)

        try:
            if embedding is not None:
                # If the TTS supports speaker_vector or similar, pass it
                try:
                    tts.tts_to_file(text=text, file_path=output_path, speaker_vector=embedding)
                    return
                except TypeError:
                    pass

            # Fallbacks: try speaker_wav directly if embedding not used
            if speaker_ref and Path(speaker_ref).exists():
                try:
                    tts.tts_to_file(text=text, file_path=output_path, speaker_wav=str(speaker_ref))
                    return
                except TypeError:
                    pass

            # Final fallback: plain TTS
            tts.tts_to_file(text=text, file_path=output_path)
        except Exception as e:
            print(f"TTS generation failed: {e}")
            raise

    def sanity_check(self):
        """Run lightweight checks for required components (Whisper/TTS availability).

        Returns a small dict with status info. This is safe to run without heavy downloads.
        """
        report = {"whisper": False, "tts": False, "models_dir": str(self.model_dir)}
        try:
            # Whisper load may download; attempt a tiny model if available
            model_size = os.getenv("WHISPER_MODEL", "tiny")
            whisper.load_model(model_size)
            report["whisper"] = True
        except Exception as e:
            report["whisper_error"] = str(e)

        try:
            tts = self.load_tts()
            report["tts"] = True
        except Exception as e:
            report["tts_error"] = str(e)

        return report

        # After generation, attempt to align duration to original audio length in downstream step
    
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

            # Align generated audio duration to the original audio duration.
            try:
                from pydub import AudioSegment

                orig = AudioSegment.from_file(str(audio_path))
                gen = AudioSegment.from_file(str(translated_audio_path))

                orig_ms = len(orig)
                gen_ms = len(gen)

                if gen_ms == 0:
                    raise ValueError("Generated TTS audio is empty")

                if abs(gen_ms - orig_ms) > 50:  # if difference greater than 50ms, adjust
                    print(f"Adjusting generated audio from {gen_ms}ms -> {orig_ms}ms to match original")
                    # Simple approach: speed change factor
                    speed = gen_ms / orig_ms
                    # Change speed by resampling
                    new_frame_rate = int(gen.frame_rate / speed)
                    stretched = gen._spawn(gen.raw_data, overrides={"frame_rate": new_frame_rate})
                    stretched = stretched.set_frame_rate(gen.frame_rate)
                    # Trim or pad to exact length
                    if len(stretched) > orig_ms:
                        stretched = stretched[:orig_ms]
                    else:
                        silence = AudioSegment.silent(duration=orig_ms - len(stretched))
                        stretched = stretched + silence

                    # overwrite translated_audio_path
                    stretched.export(str(translated_audio_path), format="wav")
            except Exception as e:
                print(f"Audio alignment step skipped/failed: {e}")
            
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
    
    # Attempt to pre-download a multilingual model to support more target languages.
    try:
        print("Pre-downloading multilingual m2m100 model for broader language coverage...")
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        AutoTokenizer.from_pretrained("facebook/m2m100_418M")
        AutoModelForSeq2SeqLM.from_pretrained("facebook/m2m100_418M")
    except Exception as e:
        print(f"Could not pre-download m2m100 model: {e}")
    
    # TTS model (base)
    TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False)

    # Note: Wav2Lip repository and checkpoint are not automatically cloned by this script.
    # For full lip-sync install, clone Wav2Lip into the models directory and place the
    # checkpoint at models/Wav2Lip/checkpoints/wav2lip_gan.pth
    print("If you need full Wav2Lip lip-syncing, clone the Wav2Lip repo into the models folder and add the checkpoint file.")
    
    print("Models downloaded successfully!")

if __name__ == "__main__":
    download_models()