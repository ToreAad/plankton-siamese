# Siamese plankton

0. Install necessary software:

   pip install keras
   pip install tensorflow=1.5 (our CPU doesn't support AVX instructions)
   sudo apt-get install imagemagick

1. Use imagemagick to rescale all images. Images smaller than 299x299
are centered on a white background, larger images are scaled down to
fit.  This preserves size for small images.

  (script: convert-images-sh)

2. Select classes to learn (more than xxx objects)

Split into random subsets, 100 validate, 100 test, rest in train

  (script: prepare-data.sh)
  parallel echo {} \| bash prepare-data.sh ::: $(ls data)
 

3. Run training

   (create-model.py, train.py)

4. Test performance

a) confusion matrix
b) error examples
