import os
import pretty_midi
import json
import random

# Define paths
DATASET_PATH = 'xmidi_dataset'
METADATA_FILE = 'xmidi_metadata.json'

# Genre and emotion options
GENRES = ['Classical', 'Jazz']
EMOTIONS = ['Happy', 'Angry']

# Number of MIDI files to process for a smaller dataset
MAX_FILES = 941

# Generate metadata dictionary
metadata = {}

# Get list of MIDI files and limit to MAX_FILES
midi_files = [file for file in os.listdir(DATASET_PATH) if file.endswith('.mid') or file.endswith('.midi')]
midi_files = random.sample(midi_files, min(len(midi_files), MAX_FILES))

# Scan and process selected MIDI files
for file in midi_files:
    genre = random.choice(GENRES)
    emotion = random.choice(EMOTIONS)

    # Add metadata for the file
    metadata[file] = {
        'genre': genre,
        'emotion': emotion
    }

# Save metadata as JSON
with open(METADATA_FILE, 'w') as f:
    json.dump(metadata, f, indent=4)

print(f'Metadata saved as {METADATA_FILE}')
