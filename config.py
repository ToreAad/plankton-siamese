# Input data parameters
train_dir = 'train'
val_dir = 'validate'
test_dir = 'test'
in_dim = (299, 299, 1)
out_dim = 64
n_classes = 40

# Base network parameters
base_model = "inception"#"simple_convolutional"
base_batch_size = 20
base_steps_per_epoch = 1000
base_validation_steps = 100
base_epochs = 25
base_learn_rate = 0.01

# Siamese network parameters
siamese_batch_size = 20
siamese_steps_per_epoch = 1000
siamese_validation_steps = 100
siamese_epochs = 25
logfile = 'train.log'
learn_rate = 0.01
lr_decay = 0.9
last = 0

plankton_str2int = {
"Annelida" : 0,
"Bivalvia__Mollusca" : 1,
"Brachyura" : 2,
"calyptopsis" : 3,
"Candaciidae" : 4,
"Cavoliniidae" : 5,
"Centropagidae" : 6,
"Corycaeidae" : 7,
"Coscinodiscus" : 8,
"cyphonaute" : 9,
"Decapoda" : 10,
"Doliolida" : 11,
"egg__Actinopterygii" : 12,
"egg__other" : 13,
"Eucalanidae" : 14,
"Euchaetidae" : 15,
"eudoxie__Diphyidae" : 16,
"Evadne" : 17,
"Foraminifera" : 18,
"Fritillariidae" : 19,
"gonophore__Diphyidae" : 20,
"Haloptilus" : 21,
"Harpacticoida" : 22,
"Limacinidae" : 23,
"multiple__Copepoda" : 24,
"multiple__other" : 25,
"nauplii__Cirripedia" : 26,
"nauplii__Crustacea" : 27,
"nectophore__Diphyidae" : 28,
"Noctiluca" : 29,
"Oikopleuridae" : 30,
"Oncaeidae" : 31,
"Ostracoda" : 32,
"Penilia" : 33,
"Phaeodaria" : 34,
"Salpida" : 35,
"tail__Appendicularia" : 36,
"tail__Chaetognatha" : 37,
"Temoridae" : 38,
"zoea__Decapoda" : 39
}

plankton_int2str = dict([(item, val) for val, item in plankton_str2int.items()])