import pickle
import tensorflow as tf
import pandas as pd
import numpy as np

MAX_LENGTH = 40
BATCH_SIZE = 32
BUFFER_SIZE = 1000
EMBEDDING_DIM = 512
UNITS = 512


# LOADING DATA
vocab = pickle.load(open('vocabulary/vocab_coco.file', 'rb'))

tokenizer = tf.keras.layers.TextVectorization(
    standardize = None,
    output_sequence_length = MAX_LENGTH,
    vocabulary = vocab
)

idx2word = tf.keras.layers.StringLookup(
    mask_token = "",
    vocabulary = tokenizer.get_vocabulary(),
    invert = True
)

def load_image_from_path(img_path):
    img = tf.io.read_file(img_path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.keras.layers.Resizing(299, 299)(img)
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img


def generate_caption(img, caption_model, add_noise=False):
    if isinstance(img, str):
        img = load_image_from_path(img)
    
    if add_noise == True:
        noise = tf.random.normal(img.shape)*0.1
        img = (img + noise)
        img = (img - tf.reduce_min(img))/(tf.reduce_max(img) - tf.reduce_min(img))
    
    img = tf.expand_dims(img, axis=0)
    img_embed = caption_model.cnn_model(img)
    img_encoded = caption_model.encoder(img_embed, training=False)

    y_inp = '[start]'
    for i in range(MAX_LENGTH-1):
        tokenized = tokenizer([y_inp])[:, :-1]
        mask = tf.cast(tokenized != 0, tf.int32)
        pred = caption_model.decoder(
            tokenized, img_encoded, training=False, mask=mask)
        
        pred_idx = np.argmax(pred[0, i, :])
        pred_word = idx2word(pred_idx).numpy().decode('utf-8')
        if pred_word == '[end]':
            break
        
        y_inp += ' ' + pred_word
    
    y_inp = y_inp.replace('[start] ', '')
    return y_inp


def get_caption_model():
    encoder = TransformerEncoderLayer(EMBEDDING_DIM, 1)
    decoder = TransformerDecoderLayer(EMBEDDING_DIM, UNITS, 8)

    cnn_model = CNN_Encoder()

    caption_model = ImageCaptioningModel(
        cnn_model=cnn_model, encoder=encoder, decoder=decoder, image_aug=None,
    )

    def call_fn(batch, training):
        return batch

    caption_model.call = call_fn
    sample_x, sample_y = tf.random.normal((1, 299, 299, 3)), tf.zeros((1, 40))

    caption_model((sample_x, sample_y))

    sample_img_embed = caption_model.cnn_model(sample_x)
    sample_enc_out = caption_model.encoder(sample_img_embed, training=False)
    caption_model.decoder(sample_y, sample_enc_out, training=False)

    try:
        caption_model.load_weights('models/trained_coco_weights.h5')
    except FileNotFoundError:
        caption_model.load_weights('image-caption-generator/models/trained_coco_weights.h5')

    return caption_model