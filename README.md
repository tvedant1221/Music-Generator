# Music-Generator
AI Music Generator is a deep learning project that creates original music in the form of MIDI files based on two user-selected inputs: genre and emotion. Built using a lightweight LSTM-based neural network, the model is trained on a subset of the XMIDI dataset and generates melodies by predicting musical note sequences.
This project is a lightweight prototype for an **AI-powered music generation system**. It generates MIDI music based on **two factors**:  
- 🎼 **Genre**  
- 🎭 **Emotion**  

The third factor, instrument, was intentionally excluded for simplicity and performance.  

---

## 🚀 Features

- Generate new MIDI files based on genre and emotion combinations.
- Trained on a curated subset of the [XMIDI dataset](https://github.com/AI-Music-Generation/xmidi).
- LSTM-based neural network with bidirectional layers for better musical understanding.
- Simple Flask web app with a user-friendly interface.

---

## 🧠 Model Overview

- **Architecture**: Embedding → BiLSTM × 2 → LSTM → Dense
- **Input**: Sequences of note pitches
- **Output**: Probability distribution of next note
- **File**: `music_generation_model.h5`

---

## 📁 Project Structure

```plaintext
├── xmidi_dataset/              # Folder with MIDI files
├── xmidi_metadata.json         # Metadata: genre & emotion per file
├── small_midi_data.npy         # Preprocessed training sequences
├── Music Model Training.py     # Model training script
├── app.py                      # Flask web app for generation
├── templates/
│   └── index.html              # Web UI for choosing genre & emotion
└── generated_music.mid         # Output file after generation


✅ Example Usage
- Choose a genre (e.g., Jazz)
- Choose an emotion (e.g., Calm)
- Click Generate Music
- 🎧 Download the MIDI file and play it using VLC, GarageBand, or any DAW
