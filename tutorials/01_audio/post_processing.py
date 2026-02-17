"""
Phone Audio Simulation Module

This module provides a research-grade simulation of phone audio characteristics,
including Impulse Response (IR) convolution, noise addition, and soft saturation.
It is designed to mimic the physical processes of capturing audio with a phone microphone.

Based on the "simulate_phone_recording.py" script.

Usage:
    from post_processing import PhoneSimulator
    
    simulator = PhoneSimulator()
    processed_audio = simulator.process(audio_data, sample_rate)
"""

import os
import re
import random
import zipfile
import requests
import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve
from pathlib import Path
from typing import Optional, List, Tuple, Union

# Default configuration for "best realism"
DEFAULT_CONFIG = {
    # Dataset
    "zenodo_record": "4633508",
    "dataset_dir": "dscaper_data/ir_dataset",
    "ir_subdir": "irs",

    # IR selection
    "min_distance_m": 1.5,  # Best realism: 1.5m
    "exclude_angle_deg": 0,
    "random_ir": True,

    # Augmentations
    "use_ir": True,
    "use_noise": True,
    "use_saturation": True,

    # Parameters
    "noise_level": 0.005,      # Increased noise
    "saturation_drive": 4.0,   # Increased saturation
    "input_gain_db": 6.0,      # Boost input signal before processing
    
    # New degradations
    "apply_bandpass": True,
    "bandpass_lo": 300,
    "bandpass_hi": 3400,
    
    "apply_downsample": True,
    "target_sample_rate": 8000,
    
    "apply_quantization": True,
    "bit_depth": 8,
    
    "apply_clipping": True,
    "clipping_threshold": 0.9,
}

class PhoneSimulator:
    def __init__(self, config: Optional[dict] = None, download: bool = True):
        """
        Initialize the PhoneSimulator with configuration.
        
        Args:
            config: Dictionary overriding default configuration.
            download: Whether to download the dataset if missing.
        """
        self.config = DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)
            
        # Resolve dataset directory relative to this file if not absolute
        dataset_dir = Path(self.config["dataset_dir"])
        if not dataset_dir.is_absolute():
            # Assume relative to this file's directory
            base_dir = Path(__file__).parent
            dataset_dir = base_dir / dataset_dir
            
        self.config["dataset_dir"] = str(dataset_dir)
        
        if download:
            self._download_dataset()
            
        self.irs = self._collect_irs()

    def _bandpass_filter(self, data: np.ndarray, sr: int, lowcut: float, highcut: float, order: int = 5) -> np.ndarray:
        """Applies a Butterworth bandpass filter."""
        from scipy.signal import butter, lfilter
        nyq = 0.5 * sr
        low = lowcut / nyq
        high = highcut / nyq
        # Ensure bounds
        low = max(0.001, min(low, 0.99))
        high = max(low + 0.001, min(high, 0.99))
        
        b, a = butter(order, [low, high], btype='band')
        y = lfilter(b, a, data)
        return y

    def _downsample_upsample(self, data: np.ndarray, original_sr: int, target_sr: int) -> np.ndarray:
        """Simulates lower sample rate by downsampling and then upsampling back."""
        if target_sr >= original_sr:
            return data
            
        from scipy.signal import resample
        num_samples_target = int(len(data) * target_sr / original_sr)
        downsampled = resample(data, num_samples_target)
        upsampled = resample(downsampled, len(data))
        return upsampled

    def _quantize(self, data: np.ndarray, bits: int) -> np.ndarray:
        """Reduces bit depth."""
        steps = 2 ** bits
        return np.round(data * (steps / 2)) / (steps / 2)

    def _hard_clip(self, data: np.ndarray, threshold: float) -> np.ndarray:
        """Hard clipping at threshold."""
        return np.clip(data, -threshold, threshold)


    def _download_dataset(self):
        """Downloads the IR dataset from Zenodo if not present."""
        dataset_dir = Path(self.config["dataset_dir"])
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # Check if IRs already exist
        ir_subdir = dataset_dir / self.config["ir_subdir"]
        if ir_subdir.exists() and any(ir_subdir.iterdir()):
            return

        print(f"Downloading dataset to {dataset_dir}...")
        api = f"https://zenodo.org/api/records/{self.config['zenodo_record']}"
        try:
            metadata = requests.get(api).json()
            
            for file in metadata.get("files", []):
                url = file["links"]["self"]
                name = file["key"]
                out = dataset_dir / name

                if out.exists():
                    continue

                print(f"Downloading {name}...")
                with requests.get(url, stream=True) as r:
                    r.raise_for_status()
                    with open(out, "wb") as f:
                        for chunk in r.iter_content(8192):
                            f.write(chunk)

                if name.endswith(".zip"):
                    print(f"Extracting {name}...")
                    with zipfile.ZipFile(out) as z:
                        z.extractall(dataset_dir / self.config["ir_subdir"])
        except Exception as e:
            print(f"Warning: Failed to download dataset: {e}")
            print("Simulation will proceed without IRs if possible, or fail if IRs are required.")

    def _parse_meta(self, name: str) -> Tuple[Optional[float], Optional[float]]:
        """Parses angle and distance from filename."""
        name = name.lower()
        a = re.search(r'(\d+)\s*deg', name)
        d = re.search(r'(\d+(\.\d+)?)\s*m', name)
        
        angle = float(a.group(1)) if a else None
        dist = float(d.group(1)) if d else None
        return angle, dist

    def _collect_irs(self) -> List[str]:
        """Collects valid IR file paths based on configuration."""
        if not self.config["use_ir"]:
            return []

        dataset_dir = Path(self.config["dataset_dir"])
        ir_root = dataset_dir / self.config["ir_subdir"]
        
        if not ir_root.exists():
            if self.config["use_ir"]:
                print(f"Warning: IR directory {ir_root} not found. Disabling IR convolution.")
                self.config["use_ir"] = False
            return []

        irs = []
        for root, _, files in os.walk(ir_root):
            for f in files:
                if not f.endswith(".wav"):
                    continue

                full_path = os.path.join(root, f)
                angle, dist = self._parse_meta(f)

                if dist is None:
                    continue
                if dist < self.config["min_distance_m"]:
                    continue
                if angle == self.config["exclude_angle_deg"]:
                    continue

                irs.append(full_path)

        if not irs and self.config["use_ir"]:
            print("Warning: No matching IRs found with current criteria. Disabling IR convolution.")
            self.config["use_ir"] = False
            
        return irs

    def _load_mono(self, path: str, target_sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
        """Loads audio file as mono, optionally resampling."""
        try:
            x, sr = sf.read(path)
        except Exception as e:
            print(f"Error reading {path}: {e}")
            return np.zeros(0), 0
            
        if x.ndim > 1:
            x = x.mean(axis=1)
            
        if target_sr and sr != target_sr:
            # Simple resampling using scipy if available
            try:
                from scipy.signal import resample
                num_samples = int(len(x) * target_sr / sr)
                x = resample(x, num_samples)
                sr = target_sr
            except ImportError:
                print("Warning: scipy not found, cannot resample IR.")
                
        return x, sr

    def _normalize_ir(self, h: np.ndarray) -> np.ndarray:
        """Normalizes Impulse Response."""
        return h / (np.sqrt(np.sum(h**2)) + 1e-12)

    def _soft_saturate(self, x: np.ndarray, drive: float) -> np.ndarray:
        """Applies soft saturation (tanh)."""
        return np.tanh(drive * x)

    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Process audio with phone simulation effects.
        
        Args:
            audio: Input audio array (numpy).
            sample_rate: Sample rate of the audio.
            
        Returns:
            Processed audio array.
        """
        y = audio.copy()
        
        # Ensure mono for processing
        if y.ndim > 1:
             y = y.mean(axis=1)

        # IR convolution
        if self.config.get("use_ir") and self.irs:
            if self.config.get("random_ir", True):
                ir_path = random.choice(self.irs)
            else:
                ir_path = self.irs[0]
                
            try:
                # Load IR and resample to match input audio sample rate
                h, ir_sr = self._load_mono(ir_path, target_sr=sample_rate)
                
                if len(h) > 0:
                    h = self._normalize_ir(h)
                    # Use fftconvolve for efficiency
                    y = fftconvolve(y, h, mode="full")
                    
                    # Optional: Trim to original length? 
                    # For phone simulation, the reverb tail is part of the effect, 
                    # but usually we want the output to be roughly the same duration 
                    # or slightly longer. "full" mode is correct for reverb.
            except Exception as e:
                print(f"Error applying IR {ir_path}: {e}")

        # Input Gain Boost (after IR loss)
        if self.config.get("input_gain_db", 0) != 0:
            gain_factor = 10 ** (self.config["input_gain_db"] / 20)
            y *= gain_factor

        # Noise
        if self.config.get("use_noise"):
            # Generate white noise
            noise_level = self.config.get("noise_level", 0.0015)
            noise = np.random.randn(len(y)) * noise_level
            y += noise

        # Bandpass Filter
        if self.config.get("apply_bandpass"):
            y = self._bandpass_filter(y, sample_rate, 
                                      self.config.get("bandpass_lo", 300), 
                                      self.config.get("bandpass_hi", 3400))

        # Downsample / Upsample
        if self.config.get("apply_downsample"):
            y = self._downsample_upsample(y, sample_rate, 
                                          self.config.get("target_sample_rate", 8000))

        # Hard Clipping
        if self.config.get("apply_clipping"):
            y = self._hard_clip(y, self.config.get("clipping_threshold", 0.9))

        # Quantization
        if self.config.get("apply_quantization"):
            y = self._quantize(y, self.config.get("bit_depth", 8))

        # Saturation
        if self.config.get("use_saturation"):
            drive = self.config.get("saturation_drive", 4.0)
            y = self._soft_saturate(y, drive)

        # Normalize output
        max_val = np.max(np.abs(y)) + 1e-9
        if max_val > 0:
            y /= max_val
        
        return y

    def process_file(self, input_path: str, output_path: str):
        """Reads, processes, and saves an audio file."""
        x, sr = self._load_mono(input_path)
        if len(x) == 0:
            return
            
        y = self.process(x, sr)
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        sf.write(output_path, y, sr)
        print(f"Processed: {input_path} -> {output_path}")

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Simulate phone recording effects.")
    parser.add_argument("--input", "-i", help="Input audio file or directory")
    parser.add_argument("--output", "-o", help="Output audio file or directory")
    parser.add_argument("--download", action="store_true", help="Download IR dataset if missing")
    
    args = parser.parse_args()
    
    if args.input and args.output:
        sim = PhoneSimulator(download=args.download)
        
        input_path = Path(args.input)
        output_path = Path(args.output)
        
        if input_path.is_dir():
            output_path.mkdir(parents=True, exist_ok=True)
            for f in input_path.glob("*.wav"):
                sim.process_file(str(f), str(output_path / f.name))
        else:
            sim.process_file(str(input_path), str(output_path))
    else:
        print("Please provide input and output paths.")
        print("Example: python post_processing.py -i input.wav -o output.wav")

# # Process a single file
# python tutorials/01_audio/post_processing.py -i input.wav -o output.wav
# python post_processing.py -i ./dialog_0074_0139/overlap_pauses_sound_effects_wet.wav -o output_dialog_0074_0139.wav

# # Process a directory of WAV files
# python tutorials/01_audio/post_processing.py -i input_dir/ -o output_dir/
