annotation_folder = '/annotations/'
image_folder = '/train2014/'
checkpoint_path = './checkpoints/train'

top_k = 5000
num_examples = 30000
# extract_images = 500

# Tunable parameters depending on the system's configuration.
BATCH_SIZE = 64
BUFFER_SIZE = 1000
embedding_dim = 256
units = 512

# Shape of the vector extracted from InceptionV3 is (64, 2048).
# It represent the vector shape.
feature_shape = 2048
attention_features_shape = 64
start_epoch = 0
