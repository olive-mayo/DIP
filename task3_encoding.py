import cv2
import os
import heapq
from collections import Counter

os.makedirs("outputs/task3", exist_ok=True)

img = cv2.imread("assets/food.jpg", cv2.IMREAD_GRAYSCALE)
flat = img.flatten()
freq = Counter(flat)

def shannon_fano(symbols):
    if len(symbols) <= 1:
        return {symbols[0][0]: '0'} if len(symbols) == 1 else {}
    total = sum([f for _, f in symbols])
    acc, split = 0, 0
    for i, (_, f) in enumerate(symbols):
        acc += f
        if acc >= total / 2:
            split = i
            break
    left = shannon_fano(symbols[:split+1])
    right = shannon_fano(symbols[split+1:])
    for k in left: left[k] = '0' + left[k]
    for k in right: right[k] = '1' + right[k]
    return {**left, **right}

sf_codes = shannon_fano(sorted(freq.items(), key=lambda x: -x[1]))
sf_encoded = ''.join([sf_codes[p] for p in flat])
with open("outputs/task3/shannon_fano.txt", "w") as f:
    f.write(sf_encoded)

class Node:
    def __init__(self, sym, freq):
        self.sym = sym; self.freq = freq; self.left = None; self.right = None
    def __lt__(self, other): return self.freq < other.freq

def huffman(freq):
    heap = [Node(s, f) for s, f in freq.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        n1, n2 = heapq.heappop(heap), heapq.heappop(heap)
        parent = Node(None, n1.freq + n2.freq); parent.left, parent.right = n1, n2
        heapq.heappush(heap, parent)
    codes = {}
    def gen(node, code=""):
        if node:
            if node.sym is not None: codes[node.sym] = code
            gen(node.left, code + "0"); gen(node.right, code + "1")
    gen(heap[0]); return codes

hf_codes = huffman(freq)
hf_encoded = ''.join([hf_codes[p] for p in flat])
with open("outputs/task3/huffman.txt", "w") as f:
    f.write(hf_encoded)
