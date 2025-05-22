import pretty_midi
import numpy as np
import os
import json
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout, Bidirectional
import random

# Define parameters
SEQ_LENGTH = 50
NUM_NOTES = 128
DATASET_PATH = '/kaggle/input/music-generation100/xmidi_dataset'
# METADATA_FILE removed since instruments are no longer used
DATA_FILE = 'small_midi_data.npy'
MODEL_FILE = 'music_generation_model.h5'

# Number of MIDI files to process for a smaller dataset
MAX_FILES = 941

# (Optional) Remove metadata loading since it's not used in training:
# with open(METADATA_FILE, 'r') as f:
#     metadata = json.load(f)


# Function to convert MIDI to sequence
def midi_to_sequence(midi_file):
    midi_data = pretty_midi.PrettyMIDI(midi_file)
    notes = []
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            notes.append(note.pitch)
    return notes


# Load and process MIDI files
def prepare_dataset():
    sequences = []
    midi_files = [file for file in os.listdir(DATASET_PATH) if file.endswith('.mid') or file.endswith('.midi')]
    midi_files = random.sample(midi_files, min(len(midi_files), MAX_FILES))  # Limit files to MAX_FILES

    for file in midi_files:
        midi_file = os.path.join(DATASET_PATH, file)
        note_sequence = midi_to_sequence(midi_file)

        # Create sequences if note length is sufficient
        if len(note_sequence) > SEQ_LENGTH:
            for i in range(len(note_sequence) - SEQ_LENGTH):
                sequences.append(note_sequence[i:i + SEQ_LENGTH + 1])

    sequences = np.array(sequences)
    np.save(DATA_FILE, sequences)
    print(f'Dataset saved as {DATA_FILE}')


# Prepare dataset
prepare_dataset()


# Load dataset
data = np.load(DATA_FILE)

# Ensure non-empty dataset
if len(data) == 0:
    raise ValueError('No valid sequences found. Please check the dataset.')

x_data = data[:, :-1]
y_data = data[:, -1]

# Normalize data
x_data = x_data / float(NUM_NOTES)
y_data = tf.keras.utils.to_categorical(y_data, num_classes=NUM_NOTES)

# Define model
model = Sequential([
    Embedding(input_dim=NUM_NOTES, output_dim=128, input_length=SEQ_LENGTH),
    Bidirectional(LSTM(256, return_sequences=True)),  # BiLSTM captures both directions
    Dropout(0.3),
    Bidirectional(LSTM(256, return_sequences=True)),
    Dropout(0.3),
    LSTM(256),
    Dense(128, activation='relu'),
    Dense(NUM_NOTES, activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(x_data, y_data, epochs=10, batch_size=128)

# Save model
model.save(MODEL_FILE)
print(f'Model saved as {MODEL_FILE}')
