import os
import json
from typing import List, Dict, Tuple
import regex as re
from collections import Counter

class Tokenizer:
    
    def __init__(self, 
            encoder: Dict | None = {}, 
            merges: List | None = {}, 
            special_tokens: Dict = {}, 
            regex: str = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        ):
        self.regex = regex
        self.pat = re.compile(self.regex)
        self.special_tokens = special_tokens # str -> int, e.g. {'<|endoftext|>': 100257}
        self.special_pattern = "(" + "|".join(re.escape(k) for k in special_tokens) + ")"
        self.encoder = encoder
        self.decoder = {v:k.encode("utf-8") for k,v in self.encoder.items()}
        self.merges = merges
        
        
    @staticmethod
    def load(root_path: str):
        encoder_path = os.path.join(root_path,"encoder.json")
        merges_path = os.path.join(root_path,"merges.bpe")
        if not os.path.exists(encoder_path):
            raise Exception(f"file {encoder_path} not found!")
        if not os.path.exists(merges_path):
            raise Exception(f"file {merges_path} not found!")
        with open(encoder_path, 'r') as f:
            encoder = json.load(f)
        version, splitter, num_special_tokens, special_tokens, merges = "","",0,{},{}
        with open(merges_path, 'r', encoding="utf-8") as f:
            version = f.readline().strip()
            splitter = f.readline().replace("splitter: ","").strip()
            num_special_tokens = int(f.readline().replace("special_tokens#:","").strip())
            for _ in range(num_special_tokens):
                special, special_idx = f.readline().strip().split()
                special_tokens[special] = int(special_idx)
            for merge_str in f:
                merge,idx = [int(t) for t in merge_str.split("->")[0].split()], int(merge_str.split("->")[1])
                merges[tuple(merge)] = idx
        tokenizer = Tokenizer(
            encoder = encoder,
            merges = merges,
            special_tokens = special_tokens,
            regex=splitter
        )
        assert "stand tall!<|endoftext|>" == tokenizer.decode(tokenizer.encode("stand tall!<|endoftext|>"))
        return tokenizer
    
    def save(self, root_path):
        encoder_path = os.path.join(root_path,"encoder.json")
        merges_path = os.path.join(root_path,"merges.bpe")
        with open(encoder_path,"w") as fp:
            json.dump({k.decode("utf-8", errors="replace"): v for k,v in self.encoder.items()}, fp)
        with open(merges_path, "w") as fp:
            fp.write(f"version#: 01 \n")
            fp.write(f"splitter: {self.regex}\n")
            fp.write(f"special_tokens#: {len(self.special_tokens)}\n")
            for special, idx in self.special_tokens.items():
                fp.write(f"{special} {idx}\n")
            for (p1,p2),idx in self.merges.items():
                fp.write(f"{p1} {p2} -> {idx}\n")
        
        
    def get_stats(self, tokens: List[int]) -> Dict[Tuple[int, int], int]:
        counts = {}
        for pair in zip(tokens, tokens[1:]):
            counts[pair] = counts.get(pair,0) + 1
        return counts
    
    def get_stats_from_token_counts(self, token_counts: Dict[str, int]):
        stats = {}
        for tokens, freq in token_counts.items():
            for pair in zip(tokens, tokens[1:]):
                stats[pair] = stats.get(pair,0) + freq
        return stats
    
    def merge(self, tokens: List[int], pair, new_id) -> List[int]:
        new_tokens = []
        i = 0
        while i < len(tokens):
            if (i+1) < len(tokens) and pair[0] == tokens[i] and pair[1] == tokens[i+1]:
                new_tokens.append(new_id)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        return new_tokens
    
    def merge_token_counts(self, token_counts: Dict, pair, new_id: int):
        new_counts = {}
        for tokens, freq in token_counts.items():
            new_tokens = self.merge(tokens, pair, new_id)
            new_counts[tuple(new_tokens)] = freq
        return new_counts
    
    def train(self, text: str, num_merges: int) -> Tuple[Dict[Tuple[int, int], int], Dict[Tuple,int]]:
        token_counts = Counter([word.encode("utf-8") for word in re.findall(self.pat,text)])
        self.merges = {}
        for i in range(num_merges):
            stats = self.get_stats_from_token_counts(token_counts)
            most_common_pair = max(stats, key=stats.get)
            new_id = 256 + i
            print(f"merging {most_common_pair} to {new_id}")
            token_counts = self.merge_token_counts(token_counts, most_common_pair, new_id)
            self.merges[most_common_pair] = new_id

        self.decoder = {idx: bytes([idx]) for idx in range(256)}
        for (p0,p1), idx in self.merges.items():
            self.decoder[idx] = self.decoder[p0] + self.decoder[p1]
        self.encoder = {v:k for k,v in self.decoder.items()}
        return self.merges, token_counts
    
    def encode(self, text: str):
        special_chunks = re.split(self.special_pattern, text)
        text_tokens = []
        for part in special_chunks:
            if part in self.special_tokens:
                text_tokens.append(self.special_tokens[part])
            else:
                for chunk in re.findall(self.pat, part):
                    tokens = list(chunk.encode("utf-8"))
                    while len(tokens) >= 2:
                        stats = self.get_stats(tokens)
                        earliest_pair = min(stats, key = lambda p: self.merges.get(p, float("inf"))) # Find the pair from my pairs that occurred earliest in the merges
                        if earliest_pair not in self.merges:
                            break
                        replacement_token_id = self.merges[earliest_pair]
                        tokens = self.merge(tokens=tokens, pair=earliest_pair, new_id=replacement_token_id)
                    text_tokens.extend(tokens)
        return text_tokens
    
    def decode(self, tokens: List[int]):
        tokens_bytes = []
        inverse_special_tokens = {v:k for k,v in self.special_tokens.items()}
        for idx in tokens:
            if idx in self.decoder:
                tokens_bytes.append(self.decoder[idx])
            elif idx in inverse_special_tokens:
                tokens_bytes.append(inverse_special_tokens[idx].encode("utf-8"))
            else:
                # raise ValueError(f"invalid token id: {idx}")
                pass
        tokens_bytes = b"".join(tokens_bytes)
        return tokens_bytes.decode("utf-8", errors="replace")