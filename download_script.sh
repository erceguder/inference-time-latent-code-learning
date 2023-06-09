#!/bin/bash

# download the FFHQ checkpoint: https://github.com/rosinality/stylegan2-pytorch#pretrained-checkpoints
# and target FFHQ-Babies dataset (10 images only)
wget https://user.ceng.metu.edu.tr/~adhd/courses/550000.pt

wget https://user.ceng.metu.edu.tr/~adhd/courses/babies_images.zip
wget https://user.ceng.metu.edu.tr/~adhd/courses/bitmoji_images.zip
wget https://user.ceng.metu.edu.tr/~adhd/courses/bitmoji_test_images.zip

unzip babies_images.zip -d babies
unzip bitmoji_images.zip -d bitmoji
unzip bitmoji_test_images.zip -d bitmoji_test

# download FFHQ-Babies full, to calculate FID scores
python -c "import gdown; url = 'https://drive.google.com/u/0/uc?id=1JmjKBq_wylJmpCQ2OWNMy211NFJhHHID'; output = 'babies_real_test.zip'; gdown.download(url, output, quiet=False)"
unzip babies_real_test.zip -d babies_test

# Resize images to output resolution
python resize.py babies_test
python resize.py bitmoji --extension jpg
python resize.py bitmoji_test --extension jpg
