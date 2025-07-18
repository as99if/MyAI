import wave
import io
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy.signal import get_window
from piper import PiperVoice, SynthesisConfig
import gradio as gr
import tempfile
import time

class TTSEngineGradio:
    def __init__(self, model_path="/home/asifahmedshuvo/Development/MyAI/my_ai_assistant/models/speech/en_GB-semaine-medium.onnx"):
        self.voice = PiperVoice.load(model_path)
        self.syn_config = SynthesisConfig(
            volume=0.5,
            length_scale=1.0,
            noise_scale=1.0,
            noise_w_scale=1.0,
            normalize_audio=False,
        )

    def synthesize(self, text):
        audio_buffer = io.BytesIO()
        with wave.open(audio_buffer, "wb") as wav_file:
            self.voice.synthesize_wav(text, wav_file=wav_file, syn_config=self.syn_config)
        audio_buffer.seek(0)
        return audio_buffer

    def get_audio_data(self, audio_buffer):
        with wave.open(audio_buffer, "rb") as wav_file:
            frames = wav_file.readframes(-1)
            sample_rate = wav_file.getframerate()
            n_channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            total_frames = wav_file.getnframes()

        if sample_width == 1:
            dtype = np.uint8
        elif sample_width == 2:
            dtype = np.int16
        elif sample_width == 4:
            dtype = np.int32
        else:
            dtype = np.float32

        audio_data = np.frombuffer(frames, dtype=dtype)
        if n_channels == 2:
            audio_data = audio_data.reshape(-1, 2)
            audio_data = np.mean(audio_data, axis=1)
        audio_data = audio_data.astype(np.float32)
        if dtype != np.float32:
            audio_data = audio_data / np.max(np.abs(audio_data))
        return audio_data, sample_rate, total_frames

    def spectrum_frames(self, audio_data, sample_rate, num_bands=32, chunk_duration=0.05):
        chunk_size = int(sample_rate * chunk_duration)
        freq_bands = np.logspace(np.log10(20), np.log10(sample_rate//2), num_bands + 1)
        band_centers = (freq_bands[:-1] + freq_bands[1:]) / 2
        total_chunks = int(np.ceil(len(audio_data) / chunk_size))
        for chunk_idx in range(total_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = start_idx + chunk_size
            if start_idx < len(audio_data):
                if end_idx > len(audio_data):
                    chunk_data = np.zeros(chunk_size)
                    remaining = len(audio_data) - start_idx
                    chunk_data[:remaining] = audio_data[start_idx:]
                else:
                    chunk_data = audio_data[start_idx:end_idx]
                windowed_data = chunk_data * get_window('hann', len(chunk_data))
                fft_data = np.abs(fft(windowed_data))[:chunk_size//2]
                freqs = np.fft.fftfreq(chunk_size, 1/sample_rate)[:chunk_size//2]
                band_levels = np.zeros(num_bands)
                for i in range(num_bands):
                    band_mask = (freqs >= freq_bands[i]) & (freqs < freq_bands[i+1])
                    if np.any(band_mask):
                        band_levels[i] = np.sqrt(np.mean(fft_data[band_mask]**2))
                if np.max(band_levels) > 0:
                    band_levels = band_levels / np.max(band_levels)
                    band_levels = np.log10(band_levels + 0.01) + 2
                    band_levels = np.clip(band_levels, 0, 1)
                else:
                    band_levels = np.zeros(num_bands)
                yield band_levels, band_centers

    def play_synthesized_audio_with_gradio_visualizer(self, text, interrupt_flag=False):
        audio_buffer = self.synthesize(text)
        audio_data, sample_rate, total_frames = self.get_audio_data(audio_buffer)
        num_bands = 32
        chunk_duration = 0.05
        chunk_size = int(sample_rate * chunk_duration)
        spoken_text = ""
        unspoken_text = text
        # Save audio to temp file for Gradio playback
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            tmp_wav.write(audio_buffer.getbuffer())
            audio_path = tmp_wav.name

        # Visualization frames
        spectrum_gen = self.spectrum_frames(audio_data, sample_rate, num_bands=num_bands, chunk_duration=chunk_duration)
        total_chunks = int(np.ceil(len(audio_data) / chunk_size))
        images = []
        for idx, (levels, band_centers) in enumerate(spectrum_gen):
            plt.figure(figsize=(10, 4))
            plt.bar(range(num_bands), levels, color='yellow', alpha=0.8)
            plt.ylim(0, 1)
            plt.xticks(
                range(0, num_bands, 4),
                [f"{int(band_centers[i])}Hz" if band_centers[i] < 1000 else f"{band_centers[i]/1000:.1f}kHz" for i in range(0, num_bands, 4)],
                rotation=45, fontsize=8
            )
            plt.title("LED Spectrum Analyzer", color='cyan')
            plt.tight_layout()
            # Save each frame as a temp PNG file and append the path
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as img_tmp:
                plt.savefig(img_tmp, format='png')
                images.append(img_tmp.name)
            plt.close()
            # Simulate interruption
            if interrupt_flag:
                char_position = int(len(text) * ((idx + 1) / total_chunks))
                spoken_text = text[:char_position].strip()
                unspoken_text = text[char_position:].strip()
                break

        return audio_path, images, spoken_text, unspoken_text

tts_engine = TTSEngineGradio()

def gradio_demo(text, interrupt_flag):
    audio_path, images, spoken, unspoken = tts_engine.play_synthesized_audio_with_gradio_visualizer(text, interrupt_flag)
    return audio_path, images, spoken, unspoken

with gr.Blocks() as demo:
    gr.Markdown("## 🎵 LED Spectrum Analyzer (Gradio Version)")
    text_input = gr.Textbox(label="Input Text", value="Welcome to the world of speech synthesis! This is a demonstration of text-to-speech with a real-time audio visualizer. You can stop playback at any time.")
    audio_output = gr.Audio(label="Synthesized Speech")
    gallery = gr.Gallery(label="Spectrum Animation (frames)")
    spoken_out = gr.Textbox(label="Spoken Text")
    unspoken_out = gr.Textbox(label="Unspoken Text")
    interrupt_flag = gr.State(False)

    with gr.Row():
        run_btn = gr.Button("Run")
        stop_btn = gr.Button("Stop")

    def run_callback(text, interrupt_flag):
        # Reset the interrupt flag before running
        interrupt_flag = False
        return gradio_demo(text, interrupt_flag) + (interrupt_flag,)

    def stop_callback(interrupt_flag):
        # Set the interrupt flag to True to stop playback
        interrupt_flag = True
        return (None, None, "", "", interrupt_flag)

    run_btn.click(
        run_callback,
        inputs=[text_input, interrupt_flag],
        outputs=[audio_output, gallery, spoken_out, unspoken_out, interrupt_flag]
    )
    stop_btn.click(
        stop_callback,
        inputs=[interrupt_flag],
        outputs=[audio_output, gallery, spoken_out, unspoken_out, interrupt_flag]
    )

if __name__ == "__main__":
    print('🎵 Starting Gradio LED Spectrum Analyzer...')
    demo.launch()