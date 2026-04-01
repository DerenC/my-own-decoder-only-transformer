import os

with open("data/tiny_shakespeare.txt", 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

char2idx = { char:idx for idx, char in enumerate(chars) }
idx2char = { idx:char for idx, char in enumerate(chars) }

encode = lambda string : [ char2idx[c] for c in string ]
decode = lambda li : ''.join([idx2char[i] for i in li])

vocab_filename = "tokeniser_playground/vocab(char_level).txt"

if not os.path.exists(vocab_filename):
    with open(vocab_filename, 'w') as f:
        for idx,char in idx2char.items():
            f.write(f"{idx} - {char}\n")
