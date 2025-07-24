import os
import librosa
import numpy as np
import pickle
import noisereduce as nr
from scipy.signal import butter, lfilter
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import gradio as gr
import soundfile as sf
import tempfile
import traceback

# ---------------- AudioProcessor ----------------
class AudioProcessor:
    def __init__(self, sample_rate=22050, n_mfcc=20):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.scaler = StandardScaler()

    def highpass_filter(self, y, cutoff=100):
        b, a = butter(1, cutoff / (0.5 * self.sample_rate), btype='high')
        return lfilter(b, a, y)

    def spectral_gating(self, y, prop_decrease=0.8):
        return nr.reduce_noise(y=y, sr=self.sample_rate, prop_decrease=prop_decrease, stationary=False)

    def denoise_pipeline(self, y):
        y = self.highpass_filter(y, cutoff=100)
        y = self.spectral_gating(y, prop_decrease=0.8)
        y, _ = librosa.effects.trim(y, top_db=30)
        return y

    def extract_features(self, y):
        n_fft = min(1024, len(y))
        n_fft = max(1, n_fft)
        mfcc = librosa.feature.mfcc(y=y, sr=self.sample_rate, n_mfcc=self.n_mfcc, n_fft=n_fft)
        max_width = mfcc.shape[1]

        if max_width < 3:
            mfcc_delta = np.zeros_like(mfcc)
            mfcc_delta2 = np.zeros_like(mfcc)
        else:
            width = min(9, max_width if max_width % 2 == 1 else max_width - 1)
            width = max(width, 3)
            mfcc_delta = librosa.feature.delta(mfcc, width=width)
            mfcc_delta2 = librosa.feature.delta(mfcc, order=2, width=width)

        spec_contrast = librosa.feature.spectral_contrast(y=y, sr=self.sample_rate, n_fft=n_fft)
        try:
            tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=self.sample_rate)
            tonnetz_mean, tonnetz_std = np.mean(tonnetz, axis=1), np.std(tonnetz, axis=1)
        except Exception:
            tonnetz_mean, tonnetz_std = np.zeros(6), np.zeros(6)

        zcr = librosa.feature.zero_crossing_rate(y)
        rms = librosa.feature.rms(y=y)

        return np.concatenate([
            np.mean(mfcc, axis=1), np.std(mfcc, axis=1),
            np.mean(mfcc_delta, axis=1), np.std(mfcc_delta, axis=1),
            np.mean(mfcc_delta2, axis=1), np.std(mfcc_delta2, axis=1),
            np.mean(spec_contrast, axis=1), np.std(spec_contrast, axis=1),
            tonnetz_mean, tonnetz_std,
            np.mean(zcr, axis=1), np.std(zcr, axis=1),
            np.mean(rms, axis=1), np.std(rms, axis=1)
        ])

    def load_audio(self, file_path):
        y, sr = librosa.load(file_path, sr=self.sample_rate)
        return y, sr

    def transform(self, X):
        return self.scaler.transform(X)

# ---------------- Load Model ----------------
processor = AudioProcessor()
classifier = keras.models.load_model("animal_sound_classifier_model.keras")

with open("scaler.pkl", "rb") as f:
    processor.scaler = pickle.load(f)
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

EXPECTED_FEATURES = 150

# ✅ Emoji map for final 13 animals
EMOJI_MAP = {
    "Aslan": "🦁",     # Displayed as Lion 🦁
    "Bear": "🐻",
    "Cat": "🐱",
    "Chicken": "🐔",
    "Cow": "🐄",
    "Dog": "🐶",
    "Dolphin": "🐬",
    "Donkey": "🐴",
    "Elephant": "🐘",
    "Frog": "🐸",
    "Horse": "🐎",
    "Monkey": "🐒",
    "Sheep": "🐑"
}

# ---------------- Prediction ----------------
def predict_animal_from_wav(file_path):
    try:
        y, sr = processor.load_audio(file_path)
        y_denoised = processor.denoise_pipeline(y)
        features = processor.extract_features(y_denoised)

        if features.shape[0] < EXPECTED_FEATURES:
            features = np.pad(features, (0, EXPECTED_FEATURES - features.shape[0]), mode='constant')
        elif features.shape[0] > EXPECTED_FEATURES:
            features = features[:EXPECTED_FEATURES]

        features = features.reshape(1, -1)
        features_scaled = processor.transform(features)
        y_pred = classifier.predict(features_scaled)[0]

        predicted_index = np.argmax(y_pred)
        predicted_animal = le.classes_[predicted_index]
        confidence = float(y_pred[predicted_index])

        # ✅ Replace Aslan with Lion in display text
        display_name = "Lion" if predicted_animal == "Aslan" else predicted_animal
        emoji = EMOJI_MAP.get(predicted_animal, "")

        # ✅ Build sorted confidence dictionary (Aslan → Lion for display only)
        confidence_dict = {
            (( "Lion" if le.classes_[i] == "Aslan" else le.classes_[i]) +
             (f" {EMOJI_MAP.get(le.classes_[i], '')}" if EMOJI_MAP.get(le.classes_[i]) else "")):
            float(y_pred[i])
            for i in range(len(le.classes_))
        }
        confidence_dict = dict(sorted(confidence_dict.items(), key=lambda x: x[1], reverse=True))

        # ✅ Low confidence message → Hide chart
        if confidence < 0.60:
            return (
                "🤔 **Hmm, I am not sure which animal this is**",
                gr.update(visible=False)  # Hides chart
            )

        return (
            f"👋 **Predicted Animal: {display_name.upper()} {emoji}**\n"
            f"🔹 **Confidence:** {confidence:.2%}",
            gr.update(value=confidence_dict, visible=True)  # Show chart with data
        )

    except Exception as e:
        print("🔥 ERROR TRACEBACK 🔥")
        print(traceback.format_exc())
        return "❌ **Error during prediction!**", gr.update(visible=False)

def predict_animal(audio_file):
    try:
        if audio_file is None:
            return "❌ **No audio file provided!**", gr.update(visible=False)

        return predict_animal_from_wav(audio_file)

    except Exception as e:
        print("🔥 ERROR TRACEBACK 🔥")
        print(traceback.format_exc())
        return "❌ **Error occurred!**", gr.update(visible=False)

# ---------------- Gradio UI ----------------
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🐾 **Animal Sound Classifier**
        🎤 **Record your voice OR upload a WAV file to classify the animal sound.**  
        *(Recorded voices are automatically saved as WAV for prediction)*  
        ---
        """
    )

    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(type="filepath", label="🎵 Record or Upload WAV File")
            predict_button = gr.Button("🔍 Classify Sound", variant="primary")

        with gr.Column():
            output_text = gr.Markdown()
            output_chart = gr.Label(num_top_classes=5, label="Confidence Scores", visible=False)

    predict_button.click(
        fn=predict_animal,
        inputs=audio_input,
        outputs=[output_text, output_chart]
    )

demo.launch()