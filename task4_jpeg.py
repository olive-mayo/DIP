import cv2
import numpy as np
import os

os.makedirs("outputs/task4", exist_ok=True)

QUALITY = 20
SUBSAMPLE_420 = True

QY = np.array([
 [16,11,10,16,24,40,51,61],
 [12,12,14,19,26,58,60,55],
 [14,13,16,24,40,57,69,56],
 [14,17,22,29,51,87,80,62],
 [18,22,37,56,68,109,103,77],
 [24,35,55,64,81,104,113,92],
 [49,64,78,87,103,121,120,101],
 [72,92,95,98,112,100,103,99]
], dtype=np.float32)

QC = np.array([
 [17,18,24,47,99,99,99,99],
 [18,21,26,66,99,99,99,99],
 [24,26,56,99,99,99,99,99],
 [47,66,99,99,99,99,99,99],
 [99,99,99,99,99,99,99,99],
 [99,99,99,99,99,99,99,99],
 [99,99,99,99,99,99,99,99],
 [99,99,99,99,99,99,99,99]
], dtype=np.float32)

def scale_table(T, q):
    q = max(1, min(100, q))
    S = 5000/q if q < 50 else 200 - 2*q
    X = np.floor((T * S + 50) / 100).astype(np.float32)
    X[X < 1] = 1
    X[X > 255] = 255
    return X

QY_s = scale_table(QY, QUALITY)
QC_s = scale_table(QC, QUALITY)

def pad_to_block(img, block=8):
    h, w = img.shape
    H = (h + block - 1) // block * block
    W = (w + block - 1) // block * block
    out = np.zeros((H, W), dtype=np.uint8)
    out[:h, :w] = img
    return out, h, w

def dct_quant(channel, QT):
    h, w = channel.shape
    out = np.zeros((h, w), np.float32)
    for y in range(0, h, 8):
        for x in range(0, w, 8):
            block = channel[y:y+8, x:x+8].astype(np.float32) - 128.0
            dct = cv2.dct(block)
            q = np.round(dct / QT)
            deq = q * QT
            rec = cv2.idct(deq) + 128.0
            out[y:y+8, x:x+8] = rec
    return np.clip(out, 0, 255).astype(np.uint8)

img = cv2.imread("assets/food.jpg")
h, w = img.shape[:2]
ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
Y, Cr, Cb = cv2.split(ycrcb)

Y_pad, Yh, Yw = pad_to_block(Y)

if SUBSAMPLE_420:
    Cr_ds = cv2.resize(Cr, (w//2, h//2), interpolation=cv2.INTER_AREA)
    Cb_ds = cv2.resize(Cb, (w//2, h//2), interpolation=cv2.INTER_AREA)

    Cr_pad, Crh, Crw = pad_to_block(Cr_ds)
    Cb_pad, Cbh, Cbw = pad_to_block(Cb_ds)

    Cr_rec_small = dct_quant(Cr_pad, QC_s)[:Crh, :Crw]
    Cb_rec_small = dct_quant(Cb_pad, QC_s)[:Cbh, :Cbw]

    Cr_rec = cv2.resize(Cr_rec_small, (w, h), interpolation=cv2.INTER_LINEAR)
    Cb_rec = cv2.resize(Cb_rec_small, (w, h), interpolation=cv2.INTER_LINEAR)
else:
    Cr_pad, Crh, Crw = pad_to_block(Cr)
    Cb_pad, Cbh, Cbw = pad_to_block(Cb)

    Cr_rec = dct_quant(Cr_pad, QC_s)[:Crh, :Crw]
    Cb_rec = dct_quant(Cb_pad, QC_s)[:Cbh, :Cbw]

Y_rec = dct_quant(Y_pad, QY_s)[:Yh, :Yw]

rec = cv2.merge([Y_rec, Cr_rec, Cb_rec])
rec_bgr = cv2.cvtColor(rec, cv2.COLOR_YCrCb2BGR)
cv2.imwrite(f"outputs/task4/manual_jpeg_q{QUALITY}.png", rec_bgr)
