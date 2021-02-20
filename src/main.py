import tensorflow as tf
import time
import pickle

from constants import top_k, embedding_dim, units, BATCH_SIZE, start_epoch, attention_features_shape, \
    checkpoint_path, img_name_val_file, cap_val_file, num_epochs

from encoder import CNN_Encoder
from decoder import RNN_Decoder
from loss import optimizer, loss_function

from data_prep import DataLoader

data_getter = DataLoader()
data_getter.get_imgnames_captions()
max_length, cap_vector, tokenizer = data_getter.get_tokenizer()
dataset = data_getter.get_dataset()
img_name_train, img_name_val = data_getter.img_name_train, data_getter.img_name_val
cap_train, cap_val = data_getter.cap_train, data_getter.cap_val


with open(img_name_val_file, 'wb') as fp:
    pickle.dump(img_name_val, fp)

with open(cap_val_file, 'wb') as fp:
    pickle.dump(cap_val, fp)

vocab_size = top_k + 1
encoder = CNN_Encoder(embedding_dim)
decoder = RNN_Decoder(embedding_dim, units, vocab_size) # (256, 512, 5001)
EPOCHS = num_epochs
num_steps = len(img_name_train)// BATCH_SIZE

@tf.function
def train_step(img_tensor, target):
    loss = 0
    # initializing the hidden state for each batch
    # because the captions are not related from image to image.

    hidden = decoder.reset_state(batch_size=target.shape[0])  # shape: (bs, 512)
    dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * target.shape[0], 1)  # shape: (bs,1)

    with tf.GradientTape() as tape:
        features = encoder(img_tensor)  # features shape: (bs, 64, 256)

        for i in range(1, target.shape[1]):
            # passing the features through the decoder.
            predictions, hidden, _ = decoder(dec_input, features, hidden)

            loss += loss_function(target[:, i], predictions)

            # using teacher forcing
            dec_input = tf.expand_dims(target[:, i], 1)

    total_loss = (loss / int(target.shape[1]))

    trainable_variables = encoder.trainable_variables + decoder.trainable_variables

    gradients = tape.gradient(loss, trainable_variables)

    optimizer.apply_gradients(zip(gradients, trainable_variables))

    return loss, total_loss

ckpt = tf.train.Checkpoint(encoder=encoder,
                           decoder=decoder,
                           optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

if ckpt_manager.latest_checkpoint:
    start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
    # restoring the latest checkpoint in checkpoint_path
    ckpt.restore(ckpt_manager.latest_checkpoint)

loss_plot = []
for epoch in range(start_epoch, EPOCHS):
    start = time.time()
    total_loss = 0

    for (batch, (img_tensor, target)) in enumerate(dataset):
        batch_loss, t_loss = train_step(img_tensor, target)
        total_loss += t_loss

        if batch % 100 == 0:
            print('Epoch {} Batch {} Loss {:.4f}'.format(
                epoch + 1, batch, batch_loss.numpy() / int(target.shape[1])))

    # storing the epoch end loss value to plot later.
    loss_plot.append(total_loss / num_steps)

    if epoch % 5 == 0:
        ckpt_manager.save()

    print('Epoch {} Loss {:.6f}'.format(epoch + 1, total_loss / num_steps))
    print('Time taken for 1 epoch {} second \n'.format(time.time() - start))