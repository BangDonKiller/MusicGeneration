from music21 import corpus, note, converter, chord
import json
import os

import torch
import torch.nn.functional as F


# 加載特定作曲家的所有作品
def choose_composer(composer_name):
    all_scores = corpus.getComposer(composer_name)
    return all_scores


def is_single_note_sequence(score):
    """
    判斷一首樂曲是否只包含單音符序列
    """
    for element in score.flatten().notes:
        # 如果音符不是 note.Note，說明有其他元素（例如和弦）
        if not isinstance(element, note.Note):
            return False
    return True


# 遍歷所有樂譜，找到只包含單音符的樂曲
def find_single_note_songs(all_scores):
    single_note_songs = []
    for work in all_scores:
        try:
            score = corpus.parse(work)
            if is_single_note_sequence(score):
                single_note_songs.append(work)
        except Exception as e:
            print(f"Error parsing {work}: {e}")
    return single_note_songs


# 將 single_note_songs 中的樂曲保存至指定資料夾
def save_single_note_songs(single_note_songs, output_dir):
    for work in single_note_songs:
        try:
            score = corpus.parse(work)
            file_name = os.path.basename(work).split(".")[0]
            file_path = os.path.join(output_dir, f"{file_name}.xml")
            score.write("musicxml", file_path)
            print(f"Saved {file_name} to {file_path}")
        except Exception as e:
            print(f"Error saving {work}: {e}")

    print("只包含單音符的歌曲已保存。")


def get_symbol(path):
    symbol = ["/"] * SEQUENCE

    # traverse the file in the path
    for file in os.listdir(path):
        if file.endswith(".xml"):
            score = converter.parse(os.path.join(path, file))
            for element in score.recurse():
                time_step = int(element.duration.quarterLength / step)
                for i in range(time_step):
                    if isinstance(element, note.Note):
                        symbol.append(element.pitch.midi)
                    elif isinstance(element, note.Rest):
                        symbol.append("r")
                    elif isinstance(element, chord.Chord):
                        symbol.append(element.pitches[0].midi)
            symbol.append(new_song_delimiter)
    symbol = " ".join(map(str, symbol))

    return symbol


def save_symbol(symbol, path):
    with open(path, "w") as f:
        f.write(symbol)


def create_mappings_to_json(mappings_path):
    mappings = {}

    for i in range(128):
        mappings[str(i)] = i

    mappings["r"] = 128
    mappings["/"] = 129

    # save mappings to a file
    with open(mappings_path, "w") as f:
        json.dump(mappings, f, indent=4)


def convert_song_to_int(single_file_path, mappings_path):
    # read mappings
    with open(mappings_path, "r") as f:
        mappings = json.load(f)

    # read bach.txt
    with open(single_file_path, "r") as f:
        symbol = f.read()

    # split the space
    symbol = symbol.split()

    # convert the symbol to int
    symbol = [mappings[s] for s in symbol]

    return symbol


def generate_training_sequences(symbol, sequence_length):
    # create input sequences and the corresponding outputs
    inputs = []

    for i in range(0, len(symbol) - sequence_length + 1, 32):
        sequence = symbol[i : i + sequence_length]
        inputs.append(sequence)

    # one hot encode inputs
    inputs = F.one_hot(torch.tensor(inputs), num_classes=130).float()

    inputs = inputs.numpy()
    print("inputs shape:", inputs.shape)

    train_amount = 4992
    # train_amount = 1024

    test_samples = inputs[train_amount:]
    inputs = inputs[:train_amount]

    print("輸入資料已處理完成 : ")
    print("inputs shape:", inputs.shape)
    print("test samples shape:", test_samples.shape)

    return inputs, test_samples


CREATE_DATASET = False
step = 0.25
SEQUENCE = 32
symbol = []
path = "./dataset/bach"
single_file_path = "./dataset/bach/bach.txt"
mappings_path = "./dataset/bach/mappings.json"

new_song_delimiter = "/ " * SEQUENCE

if CREATE_DATASET:
    all_scores = corpus.getComposer("bach")

    single_note_songs = find_single_note_songs(all_scores)

    # 創建保存單音符歌曲的資料夾
    output_dir = path
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    save_single_note_songs(single_note_songs, output_dir)

    symbol = get_symbol(path)

    # save symbol to a file
    save_symbol(symbol, single_file_path)

    create_mappings_to_json(mappings_path)

symbol = convert_song_to_int(single_file_path, mappings_path)

inputs, test_samples = generate_training_sequences(symbol, SEQUENCE)
