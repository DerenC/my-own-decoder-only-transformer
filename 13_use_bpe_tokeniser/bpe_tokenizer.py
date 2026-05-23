from collections import defaultdict
from typing import Dict, List, Set, Tuple
import copy

END_OF_WORD = "</w>"
VOCAB_SIZE_MULTIPLIER = 2
# 2 means increase by 100%
assert VOCAB_SIZE_MULTIPLIER > 1, "VOCAB_SIZE_MULTIPLIER should be more than 1"

'''
Conceptual rule:

1. The newly formed token will not be across a single word
    - It will either be a subword or maximally the whole word

'''

### DEFINITION OF FUNCTIONS ###

def get_vocab_set(single_text: str) -> Set[str]:
    vocabs = set()
    for char in single_text:
        vocabs.add(char)
    vocabs.add(END_OF_WORD)
    return vocabs


# word_chars_li means when all the chars in a word are placed in a tuple
# Result: A dict with word_chars_li as key & the word freq as value
# Each word_chars_li will end with an END_OF_WORD
def get_word_chars_li_freqs(single_text: str) -> Dict[Tuple[str], int]:
    word_chars_li_freqs = defaultdict(int)
    for word in single_text.split():
        word_chars_li = tuple(list(word) + [END_OF_WORD])
        word_chars_li_freqs[word_chars_li] += 1
    return word_chars_li_freqs


# From the subword_li_freqs, find those subword tuples that have not merged into a full word yet
# For each consecutive subword pair in each tuple, tabulate the frequency of the pair
# Use that tabulated result to find the most frequently appeared subword pair
# WORDING: In this code, subword_pair means a pair of consecutive subwords
def get_most_freq_subword_pair(subword_li_freqs):

    # First tabulate the freqs of consecutive subword pairs
    subword_pair_stats: Dict[tuple[str, str], int] = defaultdict(int)
    for subword_li, freq in subword_li_freqs.items():
        for i in range(len(subword_li) - 1):
            curr_subword, next_subword = subword_li[i], subword_li[i+1]
            subword_pair = (curr_subword, next_subword)
            subword_pair_stats[subword_pair] += freq

    most_freq_subword_pair = max(subword_pair_stats, key=subword_pair_stats.get) # type: ignore
    return most_freq_subword_pair

# Update subword_tup_freqs by merginr the two old subwords in merging_subwords_pair
def update_subword_tup_freqs(subword_tup_freqs, merging_subwords_pair):
    new_subword = merging_subwords_pair[0] + merging_subwords_pair[1]

    new_subword_tup_freqs = defaultdict(int)
    for subword_tup, freq in subword_tup_freqs.items():
        subword_li = []
        i = 0
        while i < len(subword_tup):

            curr_subword = subword_tup[i]

            # Add the new subword (merging the previous two old consecutive subword)
            if i < len(subword_tup) - 1 and \
                curr_subword == merging_subwords_pair[0] and \
                subword_tup[i+1] == merging_subwords_pair[1]:

                subword_li.append(new_subword)
                i += 2  # Skip by 1 since it consists of two old tokens

            # Add the old subword in as usual
            else:
                subword_li.append(curr_subword)
                i += 1

        new_subword_tup_freqs[tuple(subword_li)] = freq
    return new_subword_tup_freqs


class TrieNode:
    def __init__(self):
        self.children: dict[str, "TrieNode"] = {}
        self.token_id: int | None = None   # set only at a terminal node
        self.is_terminal: bool = False


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, token: str, token_id: int) -> None:
        node = self.root
        for ch in token:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        node.is_terminal = True
        node.token_id = token_id

    def encode(self, text: str) -> list[int]:
        """
        Greedy longest-match scan.
        At each position, descend the Trie as far as possible
        while remembering the last valid token we passed through.
        """
        ids: list[int] = []
        i = 0

        while i < len(text):
            node = self.root
            last_match_end = -1
            last_match_id = None

            j = i
            while j < len(text) and text[j] in node.children:
                node = node.children[text[j]]
                j += 1
                if node.is_terminal:        # valid token found — keep going
                    last_match_end = j
                    last_match_id = node.token_id

            if last_match_id is None:
                # No token matched — unknown character, advance one step
                raise ValueError(
                    f"Character {text[i]!r} at position {i} not in vocabulary"
                )

            ids.append(last_match_id)
            i = last_match_end             # jump past the matched token

        return ids

def get_tokeniser_trie(sorted_vocab_set: List[str]):
    trie = Trie()
    for i, vocab in enumerate(sorted_vocab_set):
        trie.insert(vocab, i)
    return trie

def invert_tokeniser(sorted_vocab_set: List[str]) -> Dict[int, str]:
    inverse_tokeniser = {}
    for token_id, vocab in enumerate(sorted_vocab_set):
        inverse_tokeniser[token_id] = vocab
    return inverse_tokeniser


### START OF TOKENISATION SCRIPT ###

text = ''
with open('data/tiny_shakespeare.txt', 'r', encoding='utf-8') as file:
    text = file.read()

vocab_set = get_vocab_set(text)
init_vocab_size = len(vocab_set)
final_vocab_size = init_vocab_size * VOCAB_SIZE_MULTIPLIER
size_inc = final_vocab_size - init_vocab_size

init_subword_li_freqs = get_word_chars_li_freqs(text)
subword_li_freqs = copy.deepcopy(init_subword_li_freqs)

merged_subword_pairs = []   # Only for INSPECTION

for _ in range(size_inc):   # Each time will just add one new vocab
    most_freq_subword_pair = get_most_freq_subword_pair(subword_li_freqs)
    subword_li_freqs = update_subword_tup_freqs(subword_li_freqs, most_freq_subword_pair)

    # merged_subword_pairs.append(most_freq_subword_pair) # Only for INSPECTION

    new_vocab = most_freq_subword_pair[0] + most_freq_subword_pair[1]
    vocab_set.add(new_vocab)

sorted_vocab_set = sorted(list(vocab_set))

tokeniser_trie = get_tokeniser_trie(sorted_vocab_set)

inverse_tokeniser = invert_tokeniser(sorted_vocab_set)

def encode(text):
    return tokeniser_trie.encode(text)

def decode(ids):
    return ''.join([inverse_tokeniser[token_id] for token_id in ids]) \
        .replace(END_OF_WORD, "")
