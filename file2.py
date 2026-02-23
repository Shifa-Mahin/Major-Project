import argparse
import os
import warnings

import joblib
import librosa
import numpy as np
import sounddevice as sd

SAMPLE_RATE = 22050
TARGET_EMOTIONS = ["HAPPY", "SAD", "ANGRY", "SURPRISE", "FEAR", "DISGUST", "NEUTRAL"]
LEGACY_4_EMOTIONS = ["HAPPY", "SAD", "ANGRY", "NEUTRAL"]

warnings.filterwarnings("ignore", message="Trying to estimate tuning from empty frequency set.")


def load_artifacts():
    try:
        model = joblib.load("speech_mood_model.pkl")
        scaler = joblib.load("scaler.pkl")
    except FileNotFoundError:
        print("[ERROR] Missing speech_mood_model.pkl or scaler.pkl")
        raise SystemExit(1)

    meta = {}
    if os.path.exists("model_meta.pkl"):
        meta = joblib.load("model_meta.pkl")
    return model, scaler, meta


def load_class_thresholds(meta):
    thresholds = {}
    if isinstance(meta, dict):
        raw = meta.get("class_thresholds", {})
        if isinstance(raw, dict):
            for k, v in raw.items():
                try:
                    thresholds[int(k)] = float(v) * 100.0
                except Exception:
                    continue
    return thresholds


def build_label_map(model, meta):
    classes = [int(c) for c in getattr(model, "classes_", [])]
    class_names = meta.get("class_names") if isinstance(meta, dict) else None

    if class_names and len(class_names) == len(classes):
        return {int(cls): str(class_names[i]).upper() for i, cls in enumerate(classes)}
    if len(classes) == 7:
        return {int(cls): TARGET_EMOTIONS[i] for i, cls in enumerate(classes)}
    if len(classes) == 4:
        return {int(cls): LEGACY_4_EMOTIONS[i] for i, cls in enumerate(classes)}
    return {int(cls): f"CLASS_{int(cls)}" for cls in classes}


def normalize_audio(audio):
    return audio / (np.max(np.abs(audio)) + 1e-8)


def preprocess_audio(audio):
    audio = audio.flatten()
    if len(audio) == 0:
        return audio
    # Remove leading/trailing silence so features focus on voiced content.
    trimmed, _ = librosa.effects.trim(audio, top_db=25)
    if len(trimmed) > int(0.2 * SAMPLE_RATE):
        audio = trimmed
    return normalize_audio(audio)


def extract_features_180(audio):
    audio = preprocess_audio(audio)
    mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=SAMPLE_RATE).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE).T, axis=0)
    return np.hstack([mfccs, chroma, mel])


def extract_features_186(audio):
    audio = preprocess_audio(audio)
    mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=SAMPLE_RATE).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE).T, axis=0)
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=SAMPLE_RATE))
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=SAMPLE_RATE))
    zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
    energy = np.sqrt(np.mean(audio**2))
    zcr_std = np.std(librosa.feature.zero_crossing_rate(audio))
    mfcc_std = np.std(mfccs)
    return np.hstack([mfccs, chroma, mel, [spectral_centroid, spectral_rolloff, zcr, energy, zcr_std, mfcc_std]])


def extract_features_266(audio):
    audio = preprocess_audio(audio)
    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=40)
    mfccs = np.mean(mfcc.T, axis=0)
    mfcc_delta = np.mean(librosa.feature.delta(mfcc).T, axis=0)
    mfcc_delta2 = np.mean(librosa.feature.delta(mfcc, order=2).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=SAMPLE_RATE).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE).T, axis=0)
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=SAMPLE_RATE))
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=SAMPLE_RATE))
    zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
    energy = np.sqrt(np.mean(audio**2))
    zcr_std = np.std(librosa.feature.zero_crossing_rate(audio))
    mfcc_std = np.std(mfccs)
    return np.hstack(
        [
            mfccs,
            chroma,
            mel,
            mfcc_delta,
            mfcc_delta2,
            [spectral_centroid, spectral_rolloff, zcr, energy, zcr_std, mfcc_std],
        ]
    )


def select_extractor(model, scaler, meta):
    if isinstance(meta, dict) and meta.get("feature_size") is not None:
        n_features = int(meta["feature_size"])
    elif getattr(scaler, "n_features_in_", None) is not None:
        n_features = int(scaler.n_features_in_)
    elif getattr(model, "n_features_in_", None) is not None:
        n_features = int(model.n_features_in_)
    else:
        n_features = 180

    if n_features == 180:
        return n_features, extract_features_180
    if n_features == 186:
        return n_features, extract_features_186
    if n_features == 266:
        return n_features, extract_features_266
    print(f"[ERROR] Unsupported feature size: {n_features}")
    raise SystemExit(1)


def predict_from_file(audio_path, min_confidence=45.0, min_margin=8.0, show_probs=True):
    if not os.path.exists(audio_path):
        print(f"[ERROR] File not found: {audio_path}")
        raise SystemExit(1)

    model, scaler, meta = load_artifacts()
    label_map = build_label_map(model, meta)
    model_classes = [int(c) for c in getattr(model, "classes_", [])]
    class_thresholds = load_class_thresholds(meta)
    n_features, extractor = select_extractor(model, scaler, meta)

    audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE)
    if len(audio) == 0:
        print("[ERROR] Empty audio file")
        raise SystemExit(1)

    proba = predict_proba_from_audio(audio, model, scaler, n_features, extractor)
    if proba is None:
        print("[ERROR] Could not extract stable features from file")
        raise SystemExit(1)
    best_label, best_conf, margin = choose_prediction(
        proba,
        model_classes,
        label_map,
        min_conf=min_confidence,
        min_margin=min_margin,
        class_thresholds=class_thresholds,
    )

    if best_label == "UNCERTAIN":
        print(f"[UNCERTAIN] Best guess confidence: {best_conf:.1f}% (margin {margin:.1f}%)")
    else:
        print(f"[PREDICTION] {best_label} ({best_conf:.1f}%)")
    if show_probs:
        print("All:", format_probabilities(proba, model_classes, label_map))


def predict_from_audio_array(
    audio,
    model,
    scaler,
    label_map,
    model_classes,
    n_features,
    extractor,
    min_confidence=45.0,
    min_margin=8.0,
    class_thresholds=None,
):
    proba = predict_proba_from_audio(audio, model, scaler, n_features, extractor)
    if proba is None:
        return None, None, None, None, "Invalid feature vector"
    best_label, best_conf, margin = choose_prediction(
        proba,
        model_classes,
        label_map,
        min_conf=min_confidence,
        min_margin=min_margin,
        class_thresholds=class_thresholds,
    )
    return best_label, best_conf, proba, margin, None


def format_probabilities(proba, model_classes, label_map):
    by_name = {}
    for idx, p in enumerate(proba):
        cls = model_classes[int(idx)]
        by_name[label_map.get(int(cls), f"CLASS_{cls}").upper()] = float(p) * 100.0
    return " | ".join([f"{emo}:{by_name.get(emo, 0.0):.1f}%" for emo in TARGET_EMOTIONS])


def choose_prediction(proba, model_classes, label_map, min_conf=45.0, min_margin=8.0, class_thresholds=None):
    ranked = np.argsort(proba)[::-1]
    best_idx = int(ranked[0])
    second_idx = int(ranked[1]) if len(ranked) > 1 else int(ranked[0])
    best_conf = float(proba[best_idx] * 100.0)
    margin = float((proba[best_idx] - proba[second_idx]) * 100.0)
    best_class = model_classes[best_idx]
    best_label = label_map.get(int(best_class), f"CLASS_{best_class}")
    class_conf_threshold = min_conf
    if class_thresholds is not None:
        class_conf_threshold = max(min_conf, float(class_thresholds.get(int(best_class), min_conf)))
    if best_conf < class_conf_threshold or margin < min_margin:
        return "UNCERTAIN", best_conf, margin
    return best_label, best_conf, margin


def predict_proba_from_audio(audio, model, scaler, n_features, extractor):
    audio = audio.flatten()
    if len(audio) == 0:
        return None

    chunk_len = int(1.0 * SAMPLE_RATE)
    hop_len = int(0.5 * SAMPLE_RATE)
    probs = []
    weights = []

    for start in range(0, max(1, len(audio) - chunk_len + 1), hop_len):
        end = start + chunk_len
        if end > len(audio):
            break
        chunk = audio[start:end]
        if np.max(np.abs(chunk)) < 0.003:
            continue
        # Skip low-energy chunks that are mostly noise/silence.
        if float(np.sqrt(np.mean(chunk**2))) < 0.002:
            continue
        feats = extractor(chunk)
        if len(feats) != n_features:
            continue
        probs.append(model.predict_proba(scaler.transform([feats]))[0])
        # Weight more voiced/energetic chunks a bit higher.
        weights.append(float(np.sqrt(np.mean(chunk**2)) + 1e-8))

    if probs:
        p = np.vstack(probs)
        w = np.array(weights, dtype=np.float64)
        w = w / np.sum(w)
        return np.sum(p * w[:, None], axis=0)

    feats = extractor(audio)
    if len(feats) != n_features:
        return None
    return model.predict_proba(scaler.transform([feats]))[0]


def run_live(
    device=None,
    duration=3.5,
    min_audio_level=0.003,
    min_confidence=45.0,
    min_margin=8.0,
    show_probs=True,
):
    model, scaler, meta = load_artifacts()
    label_map = build_label_map(model, meta)
    model_classes = [int(c) for c in getattr(model, "classes_", [])]
    class_thresholds = load_class_thresholds(meta)
    n_features, extractor = select_extractor(model, scaler, meta)

    if device is not None:
        sd.default.device = device

    print("[LIVE] Press Ctrl+C to stop")
    i = 1
    smoothed_proba = None
    alpha = 0.75
    while True:
        try:
            print(f"[{i}] Speak now ({duration:.1f}s)...", end=" ", flush=True)
            audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
            sd.wait()
            audio = audio.flatten()
            peak = float(np.max(np.abs(audio)))
            if peak < min_audio_level:
                print(f"[WARN] Too quiet (peak={peak:.4f})")
                i += 1
                continue

            label, conf, proba, margin, err = predict_from_audio_array(
                audio,
                model,
                scaler,
                label_map,
                model_classes,
                n_features,
                extractor,
                min_confidence=min_confidence,
                min_margin=min_margin,
                class_thresholds=class_thresholds,
            )
            if err:
                print(f"[WARN] {err}")
                i += 1
                continue

            if smoothed_proba is None:
                smoothed_proba = proba
            else:
                smoothed_proba = alpha * smoothed_proba + (1.0 - alpha) * proba
            smoothed_proba = smoothed_proba / np.sum(smoothed_proba)
            label, conf, margin = choose_prediction(
                smoothed_proba,
                model_classes,
                label_map,
                min_conf=min_confidence,
                min_margin=min_margin,
                class_thresholds=class_thresholds,
            )

            if label == "UNCERTAIN":
                print(f"[UNCERTAIN] confidence={conf:.1f}% margin={margin:.1f}%")
            else:
                print(f"[PREDICTION] {label} ({conf:.1f}%)")
            if show_probs:
                print("All:", format_probabilities(smoothed_proba, model_classes, label_map))
            i += 1
        except KeyboardInterrupt:
            print("\n[STOP] Live mode ended.")
            break


def main():
    parser = argparse.ArgumentParser(description="Speech emotion prediction (file or live)")
    parser.add_argument("--audio-file", help="Path to .wav/.mp3 audio file")
    parser.add_argument("--live", action="store_true", help="Use live microphone mode")
    parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
    parser.add_argument("--device", type=int, help="Audio input device index for live mode")
    parser.add_argument("--duration", type=float, default=3.5, help="Live recording duration in seconds")
    parser.add_argument("--min-audio-level", type=float, default=0.003, help="Minimum peak level for live mode")
    parser.add_argument("--min-confidence", type=float, default=45.0, help="Minimum confidence for hard label")
    parser.add_argument("--min-margin", type=float, default=8.0, help="Minimum top-2 margin for hard label")
    parser.add_argument("--hide-probs", action="store_true", help="Hide all emotion percentages in output")
    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    if args.live:
        run_live(
            device=args.device,
            duration=args.duration,
            min_audio_level=args.min_audio_level,
            min_confidence=args.min_confidence,
            min_margin=args.min_margin,
            show_probs=not args.hide_probs,
        )
        return

    if args.audio_file:
        predict_from_file(
            args.audio_file,
            min_confidence=args.min_confidence,
            min_margin=args.min_margin,
            show_probs=not args.hide_probs,
        )
        return

    run_live(
        device=args.device,
        duration=args.duration,
        min_audio_level=args.min_audio_level,
        min_confidence=args.min_confidence,
        min_margin=args.min_margin,
        show_probs=not args.hide_probs,
    )


if __name__ == "__main__":
    main()
