import os
import music21 as m21
import json
import tensorflow as tf
import numpy as np

KERN_DATASET_PATH = './deutschl/erk'
SAVE_DIR = 'dataset'
SINGLE_FILE_DATASET = 'file_dataset'
MAPPING_PATH = 'mapping.json'
SEQUENCE_LENGTH = 64
ACCEPTABLE_DURATIONS = [0.25,0.5,0.75,1.0,1.5,2.0,3.0,4.0]

def load_songs_in_kern(dataset_path: str) -> list[m21.stream.Score]:
    # go through all the files in the ds and load them with music21
    
    songs = []
    
    for path, subdirs, files in os.walk(dataset_path):
        for file in files:
            if str(file).endswith('.krn'):
                song = m21.converter.parse(os.path.join(path, file))
                songs.append(song)
    return songs
 

def load_song_in_kern(krn_file: str) -> m21.stream.Score:
    return m21.converter.parse(krn_file)

def has_acceptable_duration(song: m21.stream.Score, acceptable_duration: list[float]) -> bool:
    for note in song.flat.notesAndRests:
        if note.duration.quarterLength not in acceptable_duration:
            return False
    return True
 
def transpose(song: m21.stream.Score) -> m21.stream.Score:
    # get key from the song
    parts = song.getElementsByClass(m21.stream.Part)
    measures_part0 = parts[0].getElementsByClass(m21.stream.Measure)
    key = measures_part0[0][4]
    
    # estimate key using music21
    if not isinstance(key, m21.key.Key):
        key = song.analyse('key')
    
    # get interval for transposition, e.g., Bmaj -> Cmaj
    if key.mode == 'major':
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch('C'))
    elif key.mode == 'minor':
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch('A'))

    # transpose song by calculated interval
    transpose_song = song.transpose(interval)
    
    return transpose_song
 
def encode_song(song : m21.stream.Score, time_step: float = 0.25) -> str:
    # p = 60, d = 1.0 -> [60, "_", "_", "_"]
    
    encoded_song = []
    
    for event in song.flat.notesAndRests:
        
        # handle notes
        if isinstance(event, m21.note.Note):
            symbol = event.pitch.midi # e.g. 60
        
        #handle rests
        elif isinstance(event, m21.note.Rest):
            symbol = 'r'
        
        # convert the note/rest into time series notation
        steps = int(event.duration.quarterLength / time_step)
        for step in range(steps):
            if step == 0:
                encoded_song.append(symbol)
            else:
                encoded_song.append('_')
                
    encoded_song = ' '.join(map(str, encoded_song))
    
    return encoded_song
 
def preprocess(dataset_path: str, save_dir: str):

    # load the folk songs
    print('Loading songs...')
    songs = load_songs_in_kern(dataset_path)
    print(f'Loaded {len(songs)} songs.')
    
    for i, song in enumerate(songs):
    
        # filter out songs that have non-acceptalbe durations
        if not has_acceptable_duration(song, ACCEPTABLE_DURATIONS):
            continue
        
        # transpose songs to Cmaj/Amin
        song = transpose(song)
        
        # encode songs with music time series representation
        encoded_song = encode_song(song)
        
        # save songs to text file
        save_path = os.path.join(save_dir, str(i))
        
        with open(save_path, 'w') as fp:
            fp.write(encoded_song)

def load(file_path):
    with open(file_path, 'r') as fp:
        song = fp.read()
    return song

def create_single_file_dataset(data_set_path: str, file_data_set_path: str, sequence_length: int) -> str:
    
    new_song_delimiter = '/ ' * sequence_length
    songs = ''
    
    # load encoded songs and add delimters
    for path, _, files in os.walk(data_set_path):
        for file in files:
            file_path = os.path.join(path, file)
            song = load(file_path)
            songs += song + ' ' + new_song_delimiter
    
    songs = songs[:-1]
    
    # save string that contain all the dataset
    with open(file_data_set_path, 'w') as fp:
        fp.write(songs)
        songs.split()    
        
    return songs

def create_mapping(songs: str, mapping_path: str) -> None:
    mappings = {}
    
    # identify the vocabulary
    songs = songs.split()
    vocabulary = list(set(songs))
    
    # create mapping
    for i, symbol in enumerate(vocabulary):
        mappings[symbol] = i
    
    # save vocabulary to a json file
    with open(mapping_path, 'w') as fp:
        json.dump(mappings, fp, indent=4)

def convert_songs_to_int(songs: str, mappings_path: str) -> list[int]:
    int_songs = []
    
    # load mappings
    with open(mappings_path, 'r') as fp:
        mappings = json.load(fp)
        
    # cast songs string to a list
    songs = songs.split()
    
    # map songs to int
    for symbol in songs:
        int_songs.append(mappings[symbol])
    
    return int_songs


def generate_training_sequences(sequence_length: int, mappings_path: str):
    # [11, 12, 13, 14, ...] -> i: [11, 12], t: 13
    
    # load songs and map the mto int
    songs = load(SINGLE_FILE_DATASET)
    int_songs = convert_songs_to_int(songs, mappings_path)
    
    # 100 symbols, 64 sl, 100 - 64 = 36
    inputs = []
    targets = []

    num_sequences = len(int_songs) - sequence_length
    for i in range(num_sequences):
        inputs.append(int_songs[i:i+sequence_length])
        targets.append(int_songs[i+sequence_length])
        
    # one-hot encoding the sequences
    # inputs: (# of sequences, sequence length, vocabulary size)
    # [ [0, 1, 2], [1, 1, 2] ] -> [ [ [ 1, 0, 0 ], [ 0, 1, 0 ], [0, 0, 1] ], [ [ 0, 1, 0 ], [ 0, 1, 0 ], [ 0, 0, 1 ] ] ]
    vocabulary_size = len(set(int_songs))
    inputs = tf.keras.utils.to_categorical(inputs, num_classes=vocabulary_size)
    targets = np.array(targets)
    
    print(f'There are {len(inputs)} sequences.')
    
    return inputs, targets

def main():
    preprocess(KERN_DATASET_PATH, SAVE_DIR)
    songs = create_single_file_dataset(SAVE_DIR, SINGLE_FILE_DATASET, SEQUENCE_LENGTH)
    create_mapping(songs, MAPPING_PATH)
    inputs, targets = generate_training_sequences(SEQUENCE_LENGTH, MAPPING_PATH)
    #a = 1
    
if __name__ == '__main__':
    main()

    