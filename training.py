import tensorflow as tf
import os
import json
import pandas as pd
import re
import numpy as np
import time
import matplotlib.pyplot as plt
import collections
import random
import requests
import json
import pickle
from math import sqrt
from PIL import Image
from tqdm.auto import tqdm

from model import CNN_Encoder, TransformerEncoderLayer, Embeddings, TransformerDecoderLayer, ImageCaptioningModel

DATASET_PATH = "coco2017"
MAX_LENGTH = 40
MAX_VOCABULARY = 12000
BATCH_SIZE = 64
BUFFER_SIZE = 1000
EMBEDDING_DIM = 512
UNITS = 512
EPOCHS = 1

with open(f"{DATASET_PATH}/annotations/captions_train2017.json", "r") as f:
    data = json.load(f)
    data = data["annotations"]

img_cap_pairs = []

for sample in data:
    img_name = "%012d.jpg" % sample["image_id"]
    img_cap_pairs.append([img_name, sample["caption"]])

captions = pd.DataFrame(img_cap_pairs, columns=["image", "caption"])
captions["image"] = captions["image"].apply(lambda x: f"{DATASET_PATH}/train2017/{x}")
captions = captions.sample(70000)
captions = captions.reset_index(drop=True)
captions.head()


def preprocessing(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub("\s+", " ", text)
    text = text.strip()
    text = "[start] " + text + " [end]"
    return text


captions["caption"] = captions["caption"].apply(preprocessing)
captions.head()

tokenizer = tf.keras.layers.TextVectorization(
    max_tokens=MAX_VOCABULARY, standardize=None, output_sequence_length=MAX_LENGTH
)

tokenizer.adapt(captions["caption"])

pickle.dump(
    tokenizer.get_vocabulary(),
    open("./vocabulary/vocab_coco.file", "wb"),
)

word2idx = tf.keras.layers.StringLookup(
    mask_token="", vocabulary=tokenizer.get_vocabulary()
)

idx2word = tf.keras.layers.StringLookup(
    mask_token="", vocabulary=tokenizer.get_vocabulary(), invert=True
)

img_to_cap_vector = collections.defaultdict(list)
for img, cap in zip(captions["image"], captions["caption"]):
    img_to_cap_vector[img].append(cap)

img_keys = list(img_to_cap_vector.keys())
random.shuffle(img_keys)

slice_index = int(len(img_keys) * 0.8)
img_name_train_keys, img_name_test_keys = (
    img_keys[:slice_index],
    img_keys[slice_index:],
)

train_img = []
train_caption = []
for imgt in img_name_train_keys:
    capt_len = len(img_to_cap_vector[imgt])
    train_img.extend([imgt] * capt_len)
    train_caption.extend(img_to_cap_vector[imgt])

test_img = []
test_caption = []
for imgtest in img_name_test_keys:
    capv_len = len(img_to_cap_vector[imgtest])
    test_img.extend([imgtest] * capv_len)
    test_caption.extend(img_to_cap_vector[imgtest])

len(train_img), len(train_caption), len(test_img), len(test_caption)


def load_data(img_path, caption):
    img = tf.io.read_file(img_path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.keras.layers.Resizing(299, 299)(img)
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    caption = tokenizer(caption)
    return img, caption


train_dataset = tf.data.Dataset.from_tensor_slices((train_img, train_caption))

train_dataset = (
    train_dataset.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE)
)

test_dataset = tf.data.Dataset.from_tensor_slices((test_img, test_caption))

test_dataset = (
    test_dataset.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE)
)

image_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomContrast(0.3),
    ]
)

encoder = TransformerEncoderLayer(EMBEDDING_DIM, 1)
decoder = TransformerDecoderLayer(EMBEDDING_DIM, UNITS, 8)

cnn_model = CNN_Encoder()
caption_model = ImageCaptioningModel(
    cnn_model=cnn_model,
    encoder=encoder,
    decoder=decoder,
    image_aug=image_augmentation,
)


cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=False, reduction="none"
)

early_stopping = tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)

caption_model.compile(optimizer=tf.keras.optimizers.Adam(), loss=cross_entropy)

history = caption_model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=test_dataset,
    callbacks=[early_stopping],
)

caption_model.save_weights("./image-caption-generator/models/trained_coco_weights_2.h5")
