import os
import zipfile
import tempfile
import numpy as np
import librosa
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# ======================
# CONFIGURATION
# ======================
BASE_DIR = Path(__file__).parent.resolve()
CLASSES = ["belly_pain", "burping", "discomfort", "hungry", "tired"]
DATA_ZIP_NAME = "infant-cry-audio-corpus.zip"
DATA_ZIP_PATH = BASE_DIR / "data" / "raw" / DATA_ZIP_NAME
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed"
MODEL_PATH = BASE_DIR / "models" / "baby_cry_model.h5"

# Create directories
PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

# Set custom temp directory with write permissions
CUSTOM_TEMP_DIR = BASE_DIR / "temp"
CUSTOM_TEMP_DIR.mkdir(exist_ok=True)
os.environ['TMPDIR'] = str(CUSTOM_TEMP_DIR)

# ======================
# CORE FUNCTIONS (UPDATED)
# ======================
def verify_dataset_structure():
    """Check if dataset structure is correct"""
    errors = []
    warnings = []
    
    if not PROCESSED_DATA_PATH.exists():
        errors.append(f"Missing processed data folder at {PROCESSED_DATA_PATH}")
        return errors, warnings
    
    for class_name in CLASSES:
        class_dir = PROCESSED_DATA_PATH / class_name
        
        if not class_dir.exists():
            errors.append(f"Missing class folder: {class_name}")
            continue
            
        audio_files = list(class_dir.glob("*.wav")) + list(class_dir.glob("*.mp3"))
        if not audio_files:
            warnings.append(f"No audio files found in {class_name}")
        else:
            try:
                y, sr = librosa.load(audio_files[0], sr=None)
                if len(y) == 0:
                    warnings.append(f"Empty audio file: {audio_files[0].name}")
            except Exception as e:
                warnings.append(f"Corrupted file {audio_files[0].name}: {str(e)}")
    
    return errors, warnings

def extract_features(file_path, max_len=300):
    """Enhanced feature extraction with better error handling"""
    try:
        y, sr = librosa.load(file_path, sr=16000, duration=5)
        
        if len(y) < sr:
            st.warning(f"Short audio file: {file_path} (only {len(y)/sr:.2f} seconds)")
            
        if np.max(np.abs(y)) < 0.01:
            st.warning(f"Quiet audio file: {file_path} (max amplitude: {np.max(np.abs(y)):.4f})")
        
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        combined = np.vstack([mfcc, delta, delta2])
        
        if combined.shape[1] < max_len:
            combined = np.pad(combined, ((0,0),(0,max_len-combined.shape[1])))
        else:
            combined = combined[:, :max_len]
            
        return combined.T
    except Exception as e:
        st.warning(f"Skipped {Path(file_path).name}: {str(e)}")
        return None

def load_data():
    """Robust data loading with detailed logging"""
    features, labels = [], []
    class_counts = []
    
    st.info("🔍 Verifying dataset structure...")
    errors, warnings = verify_dataset_structure()
    
    if errors:
        for error in errors:
            st.error(error)
        return np.array([]), np.array([])
    
    if warnings:
        for warning in warnings:
            st.warning(warning)
    
    st.info("📊 Counting audio files per class...")
    for class_idx, class_name in enumerate(CLASSES):
        class_dir = PROCESSED_DATA_PATH / class_name
        audio_files = list(class_dir.glob("*.wav")) + list(class_dir.glob("*.mp3"))
        class_counts.append(len(audio_files))
        st.write(f"• {class_name}: {len(audio_files)} files")
    
    if sum(class_counts) == 0:
        st.error("❌ No audio files found in any class folder!")
        return np.array([]), np.array([])
    
    max_samples = min(class_counts) if min(class_counts) > 0 else max(class_counts)
    st.info(f"⚖️ Using {max_samples} samples per class for balancing")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    total_files = sum(min(len(list((PROCESSED_DATA_PATH / cls).glob("*"))), max_samples) for cls in CLASSES)
    processed_files = 0
    
    for class_idx, class_name in enumerate(CLASSES):
        class_dir = PROCESSED_DATA_PATH / class_name
        audio_files = list(class_dir.glob("*.wav")) + list(class_dir.glob("*.mp3"))
        audio_files = audio_files[:max_samples]
        
        for file in audio_files:
            features_array = extract_features(str(file))
            if features_array is not None:
                features.append(features_array)
                labels.append(class_idx)
            
            processed_files += 1
            progress_bar.progress(processed_files / total_files)
            status_text.text(f"Processing {class_name}: {processed_files}/{total_files}")
    
    progress_bar.empty()
    status_text.empty()
    
    st.success(f"✅ Successfully loaded {len(features)} samples")
    return np.array(features), np.array(labels)

def create_model(input_shape, num_classes):
    """Enhanced model architecture"""
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv1D(128, 5, activation='relu'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
        weighted_metrics=['accuracy']
    )
    return model

def train_and_save_model():
    """Complete training workflow with enhanced UI"""
    try:
        st.info("🔍 Verifying dataset...")
        errors, _ = verify_dataset_structure()
        if errors:
            for error in errors:
                st.error(error)
            return False
        
        with st.spinner("📦 Loading and preprocessing data..."):
            X, y = load_data()
            
            if len(X) == 0:
                st.error("❌ No valid training data available!")
                st.info("""
                Possible solutions:
                1. Check your dataset contains WAV/MP3 files
                2. Verify folder structure matches expected classes
                3. Ensure files are valid audio
                """)
                return False
            
            class_dist = {cls: np.sum(y == i) for i, cls in enumerate(CLASSES)}
            st.bar_chart(class_dist)
            
            class_weights = compute_class_weight(
                'balanced',
                classes=np.unique(y),
                y=y
            )
            class_weights = dict(enumerate(class_weights))
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        st.info(f"🧠 Training model with {len(X_train)} samples...")
        progress_bar = st.progress(0)
        status_text = st.empty()
        epoch_text = st.empty()
        
        class TrainingCallback(tf.keras.callbacks.Callback):
            def on_epoch_begin(self, epoch, logs=None):
                epoch_text.write(f"🏃‍♂️ Epoch {epoch + 1}/50")
                
            def on_epoch_end(self, epoch, logs=None):
                progress = (epoch + 1) / 50
                progress_bar.progress(progress)
                status_text.markdown(f"""
                **Training Progress**
                - Loss: {logs['loss']:.4f}
                - Accuracy: {logs['accuracy']:.2%}
                - Val Loss: {logs['val_loss']:.4f}
                - Val Accuracy: {logs['val_accuracy']:.2%}
                """)
        
        model = create_model(X_train[0].shape, len(CLASSES))
        
        history = model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_data=(X_test, y_test),
            class_weight=class_weights,
            callbacks=[
                TrainingCallback(),
                tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=str(MODEL_PATH),
                    save_best_only=True,
                    monitor='val_accuracy'
                ),
                tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
            ],
            verbose=0
        )
        
        progress_bar.progress(1.0)
        status_text.empty()
        epoch_text.empty()
        
        model.save(str(MODEL_PATH))
        st.success(f"💾 Model saved to {MODEL_PATH}")
        
        st.subheader("Training Results")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Accuracy**")
            st.line_chart({
                "Train": history.history['accuracy'],
                "Validation": history.history['val_accuracy']
            })
        
        with col2:
            st.markdown("**Loss**")
            st.line_chart({
                "Train": history.history['loss'],
                "Validation": history.history['val_loss']
            })
        
        return True
        
    except Exception as e:
        st.error(f"❌ Training failed: {str(e)}")
        return False

def predict_audio(uploaded_file):
    """Robust prediction with proper temp file handling"""
    try:
        # Create a secure temp directory
        temp_dir = tempfile.mkdtemp(dir=str(CUSTOM_TEMP_DIR))
        temp_path = os.path.join(temp_dir, "audio_to_predict.wav")
        
        # Save the uploaded file
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Verify file was written
        if not os.path.exists(temp_path):
            st.error("Failed to save temporary audio file")
            return None, None
            
        # Extract features
        features = extract_features(temp_path)
        if features is None:
            return None, None
            
        features = np.expand_dims(features, axis=0)
        
        # Load model and predict
        model = tf.keras.models.load_model(str(MODEL_PATH))
        preds = model.predict(features, verbose=0)[0]
        
        return CLASSES[np.argmax(preds)], preds
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None
    finally:
        # Clean up temp files
        try:
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.remove(temp_path)
            if 'temp_dir' in locals() and os.path.exists(temp_dir):
                os.rmdir(temp_dir)
        except Exception as e:
            st.warning(f"Couldn't clean temp files: {str(e)}")

# ======================
# STREAMLIT UI
# ======================
st.set_page_config(page_title="Baby Cry Classifier", layout="wide", page_icon="👶")
st.title("👶 Baby Cry Classification System")

# Sidebar
with st.sidebar:
    st.header("📁 Dataset Management")
    
    if st.button("🔍 Verify Dataset Structure"):
        errors, warnings = verify_dataset_structure()
        
        if not errors and not warnings:
            st.success("✅ Dataset structure is correct!")
        else:
            if errors:
                st.error("❌ Critical issues found:")
                for error in errors:
                    st.error(error)
            if warnings:
                st.warning("⚠️ Potential issues found:")
                for warning in warnings:
                    st.warning(warning)
    
    st.header("🤖 Model Training")
    if st.button("🚂 Train New Model", help="Requires valid dataset"):
        if train_and_save_model():
            st.balloons()
            st.rerun()
    
    st.header("📊 Model Status")
    if MODEL_PATH.exists():
        st.success("✅ Model available")
        model_time = pd.to_datetime(MODEL_PATH.stat().st_mtime, unit='s')
        st.caption(f"Last trained: {model_time}")
    else:
        st.warning("⚠️ No trained model found")

# Main tabs
tab1, tab2 = st.tabs(["📊 Dataset Info", "🔮 Classifier"])

with tab1:
    st.header("Dataset Information")
    
    if st.button("🔄 Refresh Dataset Stats"):
        X, y = load_data()
        
        if len(X) > 0:
            st.success(f"Loaded {len(X)} samples")
            
            class_dist = {cls: np.sum(y == i) for i, cls in enumerate(CLASSES)}
            st.bar_chart(class_dist)
            
            st.subheader("Sample Features")
            fig, ax = plt.subplots(figsize=(10, 4))
            librosa.display.specshow(X[0].T, x_axis='time', ax=ax)
            ax.set_title("MFCC Features of First Sample")
            st.pyplot(fig)
        else:
            st.error("No data loaded!")

with tab2:
    st.header("Baby Cry Classifier")
    
    uploaded_file = st.file_uploader(
        "Upload baby cry audio (WAV/MP3)", 
        type=["wav", "mp3"],
        key="classifier_upload"
    )
    
    if uploaded_file:
        col1, col2 = st.columns(2)
        
        with col1:
            st.audio(uploaded_file)
            
            try:
                y, sr = librosa.load(uploaded_file)
                
                fig, ax = plt.subplots(2, 1, figsize=(10, 6))
                
                librosa.display.waveshow(y, sr=sr, ax=ax[0])
                ax[0].set_title("Waveform")
                
                D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
                librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=ax[1])
                ax[1].set_title("Spectrogram")
                
                plt.tight_layout()
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Couldn't display audio: {str(e)}")
        
        with col2:
            if not MODEL_PATH.exists():
                st.error("Model not trained yet!")
                st.info("Please train a model using the sidebar button")
            else:
                prediction, probs = predict_audio(uploaded_file)
                
                if prediction:
                    st.success(f"Prediction: **{prediction}**")
                    
                    prob_df = pd.DataFrame({
                        "Class": CLASSES,
                        "Probability": probs
                    }).sort_values("Probability", ascending=False)
                    
                    st.bar_chart(prob_df.set_index("Class"))
                    
                    max_prob = max(probs)
                    confidence_gauge = st.progress(int(max_prob * 100))
                    
                    if max_prob < 0.6:
                        st.warning(f"Low confidence ({max_prob:.1%})")
                        st.info("Try recording a clearer/longer sample")
                    elif max_prob < 0.8:
                        st.info(f"Moderate confidence ({max_prob:.1%})")
                    else:
                        st.success(f"High confidence ({max_prob:.1%})")
                else:
                    st.warning("Could not process the audio file")

# First-run instructions
if not DATA_ZIP_PATH.exists():
    st.warning("Initial Setup Required")
    st.markdown("""
    1. Create a folder called `data/raw/` in your project directory
    2. Place your dataset zip file there and rename it to `infant-cry-audio-corpus.zip`
    3. The expected folder structure after extraction:
    ```
    data/processed/
    ├── belly_pain/
    ├── burping/
    ├── discomfort/
    ├── hungry/
    └── tired/
    ```
    """)
    st.write(f"Current project directory: `{BASE_DIR}`")

if __name__ == "__main__":
    pass