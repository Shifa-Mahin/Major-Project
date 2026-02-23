import argparse
import os
import signal
import warnings

import joblib
import librosa
import numpy as np
import sounddevice as sd

warnings.filterwarnings("ignore")

SAMPLE_RATE = 22050
DURATION = 3
TARGET_EMOTIONS = ["HAPPY", "SAD", "ANGRY", "SURPRISE", "FEAR", "DISGUST", "NEUTRAL"]
LEGACY_4_EMOTIONS = ["HAPPY", "SAD", "ANGRY", "NEUTRAL"]

stop_requested = False


def _sigint_handler(signum, frame):
    del signum, frame
    global stop_requested
    stop_requested = True


signal.signal(signal.SIGINT, _sigint_handler)


def load_artifacts():
    try:
        model = joblib.load("speech_mood_model.pkl")
        scaler = joblib.load("scaler.pkl")
    except FileNotFoundError:
        print("[ERROR] Required files not found: speech_mood_model.pkl, scaler.pkl")
        raise SystemExit(1)

    model_meta = {}
    if os.path.exists("model_meta.pkl"):
        model_meta = joblib.load("model_meta.pkl")
    print("[OK] Model and scaler loaded successfully")
    return model, scaler, model_meta


MODEL, SCALER, MODEL_META = load_artifacts()
MODEL_CLASSES = [int(c) for c in getattr(MODEL, "classes_", [])]
PROBA_INDEX_BY_CLASS = {cls_id: i for i, cls_id in enumerate(MODEL_CLASSES)}


def build_label_map():
    classes = MODEL_CLASSES
    class_names = MODEL_META.get("class_names") if isinstance(MODEL_META, dict) else None

    if class_names and len(class_names) == len(classes):
        return {int(cls): str(class_names[i]).upper() for i, cls in enumerate(classes)}

    if len(classes) == 7:
        return {int(cls): TARGET_EMOTIONS[i] for i, cls in enumerate(classes)}

    if len(classes) == 4:
        return {int(cls): LEGACY_4_EMOTIONS[i] for i, cls in enumerate(classes)}

    return {int(cls): f"CLASS_{int(cls)}" for cls in classes}


MOOD_LABELS = build_label_map()


def missing_target_emotions():
    available = {v.upper() for v in MOOD_LABELS.values()}
    return [e for e in TARGET_EMOTIONS if e not in available]


def normalize_audio(audio):
    return audio / (np.max(np.abs(audio)) + 1e-8)


def extract_features_180(audio):
    audio = normalize_audio(audio)
    mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=SAMPLE_RATE).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE).T, axis=0)
    return np.hstack([mfccs, chroma, mel])


def extract_features_186(audio):
    audio = normalize_audio(audio)
    mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=SAMPLE_RATE).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE).T, axis=0)
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=SAMPLE_RATE))
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=SAMPLE_RATE))
    zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(audio))
    energy = np.sqrt(np.mean(audio**2))
    zcr_std = np.std(librosa.feature.zero_crossing_rate(audio))
    mfcc_std = np.std(mfccs)
    return np.hstack(
        [
            mfccs,
            chroma,
            mel,
            [spectral_centroid, spectral_rolloff, zero_crossing_rate, energy, zcr_std, mfcc_std],
        ]
    )


def expected_feature_count():
    if isinstance(MODEL_META, dict) and MODEL_META.get("feature_size") is not None:
        return int(MODEL_META["feature_size"])
    if getattr(SCALER, "n_features_in_", None) is not None:
        return int(SCALER.n_features_in_)
    if getattr(MODEL, "n_features_in_", None) is not None:
        return int(MODEL.n_features_in_)
    return 180


def get_feature_extractor():
    n = expected_feature_count()
    if n == 180:
        return n, extract_features_180
    if n == 186:
        return n, extract_features_186
    return n, None


def predict_emotion(audio, min_audio_level):
    audio = audio.flatten()
    level = float(np.max(np.abs(audio)))
    if level < min_audio_level:
        return None, None, None, level, "Too quiet"

    n_features, extractor = get_feature_extractor()
    if extractor is None:
        return None, None, None, level, f"Unsupported feature size: {n_features}"

    try:
        features = extractor(audio)
    except Exception as exc:
        return None, None, None, level, f"Feature extraction failed: {exc}"

    if features is None or len(features) != n_features:
        return None, None, None, level, "Invalid feature vector"

    scaled = SCALER.transform([features])
    proba = MODEL.predict_proba(scaled)[0]
    top_idx = int(np.argmax(proba))
    cls_id = MODEL_CLASSES[top_idx] if MODEL_CLASSES else top_idx
    confidence = float(proba[top_idx] * 100.0)
    label = MOOD_LABELS.get(int(cls_id), f"CLASS_{cls_id}")
    return label, confidence, proba, level, None


def print_banner(require_seven):
    print("\n" + "=" * 60)
    print("[MOOD] SPEECH EMOTION DETECTION")
    print("=" * 60)
    print(f"Recording: {DURATION} seconds")
    print(f"Model classes: {', '.join(MOOD_LABELS[c] for c in sorted(MOOD_LABELS))}")
    missing = missing_target_emotions()
    if missing:
        print("[WARN] Model is missing: " + ", ".join(missing))
        if require_seven:
            print("[STOP] Need a 7-emotion model to continue.")
            print("Train with labels: HAPPY, SAD, ANGRY, SURPRISE, FEAR, DISGUST, NEUTRAL")
            return False
    print("Press Ctrl+C to stop")
    return True


def live_detection(device=None, min_audio_level=0.003, uncertain_threshold=45.0, require_seven=False):
    global stop_requested

    if not print_banner(require_seven):
        return

    if device is not None:
        try:
            sd.default.device = device
            print(f"[INFO] Using audio device: {device}")
        except Exception as exc:
            print(f"[WARN] Could not set device {device}: {exc}")

    iteration = 1
    try:
        while not stop_requested:
            print(f"\n[{iteration}] SPEAK NOW ({DURATION} seconds)...", end=" ", flush=True)
            audio = sd.rec(
                int(DURATION * SAMPLE_RATE),
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
            )
            sd.wait()

            if stop_requested:
                break

            label, confidence, proba, level, err = predict_emotion(audio, min_audio_level)
            if err:
                print(f"[WARN] {err} (peak={level:.4f})")
                iteration += 1
                continue

            if confidence < uncertain_threshold:
                print(f"[UNCERTAIN] Best guess: {label} ({confidence:.1f}%)")
            else:
                print(f"[{label}] Confidence: {confidence:.1f}%")

            if proba is not None:
                ranked = np.argsort(proba)[::-1][:3]
                top3 = []
                for idx in ranked:
                    cls_id = MODEL_CLASSES[int(idx)] if MODEL_CLASSES else int(idx)
                    name = MOOD_LABELS.get(int(cls_id), f"CLASS_{cls_id}")
                    top3.append(f"{name}:{proba[int(idx)] * 100:.1f}%")
                print("Top3:", " | ".join(top3))
            iteration += 1
    except KeyboardInterrupt:
        pass
    finally:
        stop_requested = False
        print("\n[STOP] Detection ended.")


def main():
    parser = argparse.ArgumentParser(description="Live speech emotion detection")
    parser.add_argument("--device", type=int, help="Audio input device index")
    parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
    parser.add_argument("--min-audio-level", type=float, default=0.003, help="Lower this if mic is quiet")
    parser.add_argument("--uncertain-threshold", type=float, default=45.0, help="Confidence threshold")
    parser.add_argument("--require-seven", action="store_true", help="Stop unless all 7 emotions are available")
    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    live_detection(
        device=args.device,
        min_audio_level=args.min_audio_level,
        uncertain_threshold=args.uncertain_threshold,
        require_seven=args.require_seven,
    )


if __name__ == "__main__":
    main()
