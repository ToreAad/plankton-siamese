# Input data parameters
train_dir = 'train'
val_dir = 'validate'
test_dir = 'test'
in_dim = (299, 299, 1)
out_dim = 64
n_classes = 40

# Base network parameters
base_model = "simple_convolutional"
base_batch_size = 64
base_steps_per_epoch = 2048//base_batch_size
base_validation_steps = 2048//base_batch_size
base_epochs = 25

# Siamese network parameters
siamese_batch_size = 64
siamese_steps_per_epoch = 2048//base_batch_size
siamese_validation_steps = 2048//base_batch_size
siamese_epochs = 25
logfile = 'train.log'
learn_rate = 0.01
lr_decay = 0.9
iterations = 5
last = 0

