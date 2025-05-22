from flask import Flask, render_template, request, send_file
import numpy as np
import pretty_midi
import json
import random
import tensorflow as tf

app = Flask(__name__)

# Load metadata and model
METADATA_FILE = 'xmidi_metadata.json'
MODEL_FILE = 'music_generation_model.h5'
DATA_FILE = 'small_midi_data.npy'

with open(METADATA_FILE, 'r') as f:
    metadata = json.load(f)

print("Metadata loaded successfully. Checking keys...")
for file, meta in metadata.items():
    print(f"File: {file}, Keys: {meta.keys()}")

model = tf.keras.models.load_model(MODEL_FILE)
SEQ_LENGTH = 50
NUM_NOTES = 128


# Generate a MIDI file from predicted notes
def generate_midi(predicted_notes, output_file):
    midi_data = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)  # Using default instrument

    for i, note_number in enumerate(predicted_notes):
        note = pretty_midi.Note(velocity=100, pitch=note_number, start=i * 0.5, end=(i + 1) * 0.5)
        instrument.notes.append(note)

    midi_data.instruments.append(instrument)
    midi_data.write(output_file)


# Predict next notes based on seed sequence
def predict_notes(seed_sequence, num_notes=100):
    predicted_notes = []
    for _ in range(num_notes):
        x_input = np.array(seed_sequence) / float(NUM_NOTES)
        x_input = np.reshape(x_input, (1, SEQ_LENGTH))
        prediction = model.predict(x_input, verbose=0)
        next_note = np.argmax(prediction)
        predicted_notes.append(next_note)
        seed_sequence.append(next_note)
        seed_sequence = seed_sequence[1:]
    return predicted_notes


@app.route('/')
def index():
    # Pass only genres and emotions to the template
    return render_template('index.html',
                           genres=list(set([meta['genre'] for meta in metadata.values()])),
                           emotions=list(set([meta['emotion'] for meta in metadata.values()])))


@app.route('/generate', methods=['POST'])
def generate():
    genre = request.form['genre']
    emotion = request.form['emotion']  # form returns 'emotion'
    print(f"Selected genre: {genre}")
    print(f"Selected emotion: {emotion}")

    # Filter files based on user selection (only genre and emotion)
    matching_files = [file for file, meta in metadata.items() if
                      meta['genre'].lower() == genre.lower() and meta['emotion'].lower() == emotion.lower()]
    print(f"Matching files: {matching_files}")

    if not matching_files:
        return "No matching files found for the selected criteria."

    selected_file = random.choice(matching_files)
    seed_sequence = np.load(DATA_FILE)[0][:SEQ_LENGTH].tolist()

    # Generate predicted notes
    predicted_notes = predict_notes(seed_sequence)
    output_file = 'generated_music.mid'
    generate_midi(predicted_notes, output_file)

    return send_file(output_file, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
