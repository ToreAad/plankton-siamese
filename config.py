# Input data parameters
train_dir = 'train'
val_dir = 'validate'
test_dir = 'test'
in_dim = (299, 299, 1)
out_dim = 64
n_classes = 40

# Base network parameters
base_model = "simple_convolutional"
base_batch_size = 8
base_steps_per_epoch = 800//base_batch_size
base_validation_steps = 800//base_batch_size
base_epochs = 1

# Siamese network parameters
siamese_batch_size = 8
siamese_steps_per_epoch = 8000//base_batch_size
siamese_validation_steps = 800//base_batch_size
siamese_epochs = 10
logfile = 'train.log'
learn_rate = 0.01
lr_decay = 0.9
iterations = 5
last = 0

