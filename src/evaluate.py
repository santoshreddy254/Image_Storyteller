import tensorflow as tf
import time

from constants import top_k, embedding_dim, units, BATCH_SIZE, start_epoch, attention_features_shape, checkpoint_path
from encoder import CNN_Encoder
from decoder import RNN_Decoder
from loss import optimizer, loss_function
from util import get_Inception, load_image
from data_prep import DataLoader
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pickle

vocab_size = top_k + 1
encoder = CNN_Encoder(embedding_dim)
decoder = RNN_Decoder(embedding_dim, units, vocab_size) # (256, 512, 5001)
ckpt = tf.train.Checkpoint(encoder=encoder,
                           decoder=decoder,
                           optimizer=optimizer)

data_getter = DataLoader()
data_getter.get_imgnames_captions()
max_length, cap_vector, tokenizer = data_getter.get_tokenizer()

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

def evaluate(image):
    # Testing- Restore the checkpoint and predict- https://www.tensorflow.org/tutorials/text/nmt_with_attention

    ckpt.restore(ckpt_manager.latest_checkpoint)

    image_features_extract_model = get_Inception()
    attention_plot = np.zeros((max_length, attention_features_shape))

    hidden = decoder.reset_state(batch_size=1)

    temp_input = tf.expand_dims(load_image(image)[0], 0)
    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

    features = encoder(img_tensor_val)

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []

    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(dec_input, features, hidden)

        attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()

        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
        result.append(tokenizer.index_word[predicted_id])

        if tokenizer.index_word[predicted_id] == '<end>':
            return result, attention_plot

        dec_input = tf.expand_dims([predicted_id], 0)

    attention_plot = attention_plot[:len(result), :]
    return result, attention_plot

def plot_attention(image, result, attention_plot):
    temp_image = np.array(Image.open(image))

    fig = plt.figure(figsize=(10, 10))

    len_result = len(result)
    for l in range(len_result):
        temp_att = np.resize(attention_plot[l], (8, 8))
        ax = fig.add_subplot(len_result//2, len_result//2, l+1)
        ax.set_title(result[l])
        img = ax.imshow(temp_image)
        ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())

    plt.tight_layout()
    plt.show()

img_name_val_file = "img_name_val.txt"
cap_val_file = "cap_val.txt"
with open(img_name_val_file, 'rb') as fp:
    img_name_val = pickle.load(fp)

with open(cap_val_file, 'rb') as fp:
    cap_val = pickle.load(fp)


rid = np.random.randint(0, len(img_name_val))
image = img_name_val[rid]
real_caption = ' '.join([tokenizer.index_word[i] for i in cap_val[rid] if i not in [0]])
result, attention_plot = evaluate(image)

print ('Real Caption:', real_caption)
print ('Prediction Caption:', ' '.join(result))
plot_attention(image, result, attention_plot)
