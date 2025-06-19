from collections import Counter

class Vocabulary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.freqs = Counter()
        self.special_tokens = ["<start>","<end>","<unk>","<pad>"]

    def build_vocab(self, caption_dict, threshold=5):
        for captions in caption_dict.values():
            for caption in captions:
                if not isinstance(caption, str):
                    continue
                words = caption.lower().strip().split()
                self.freqs.update(words)

        for idx, token in enumerate(self.special_tokens):
            self.word2idx[token] = idx
            self.idx2word[idx] = token
            idx += 1

        idx = len(self.special_tokens)

        for word, freq in self.freqs.items():
            if freq >= threshold:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1


    def __getitem__(self, word):
        return self.word2idx.get(word, self.word2idx["<unk>"])

    def __len__(self):
        return len(self.word2idx)
    
    def idx_to_word(self, index):
        return self.idx2word.get(index, "<unk>")
    
    def decode_caption(self, token_ids):
        words = []

        for token_id in token_ids:
            #token_id = token_id.item()
            word = self.idx_to_word(token_id)

            if word == "<end>":
                break

            if word not in ["<start>", "<pad>"]:
                words.append(word)

        caption = " ".join(words)
        return caption
    
    def decode_captions_debugger(self, token_ids):
        words = []

        for token_id in token_ids:
            #token_id = token_id.item()
            word = self.idx_to_word(token_id)
            words.append(word)
        return words