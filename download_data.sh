#!/bin/bash

# @TODO: fill it with your kaggle api token
#export KAGGLE_USERNAME="" KAGGLE_KEY=""

gdown -q 'https://drive.google.com/u/0/uc?id=1PQutd-JboOCOZqmd95XWxWrO8gGEvRcO' & #&export=download
gdown -q 'https://drive.google.com/u/0/uc?id=1Uu5y_y8Rjxbj2VEzvT3aBHyn4pltFgyX' & #&export=download
gdown -q 'https://drive.google.com/u/0/uc?id=1JmjKBq_wylJmpCQ2OWNMy211NFJhHHID' &
gdown -q 'https://drive.google.com/uc?id=1G_tqSI6RFZ-FQS3aFPTHblfkifNFx8oo'     &
gdown -q 'https://drive.google.com/u/0/uc?id=1zy68kLcYzlfMD3X2vonHH6YiwOyA1WXO' &

kaggle datasets download -q -d mostafamozafari/bitmoji-faces &
wget --quiet http://mmlab.ie.cuhk.edu.hk/archive/sketchdatabase/CUHK/training_88/Cropped_Images/CUHK_training_cropped_sketches.zip &

wait < <(jobs -p)

mkdir -p data
unzip -q babies_real_test.zip -d babies_test &
unzip -q bitmoji-faces.zip -d data &
unzip -q sunglasses_real_test.zip -d sunglasses_test &
unzip -q CUHK_training_cropped_sketches.zip -d data &
unzip -q genda_samples.zip &
unzip -q ckpts.zip &

wait < <(jobs -p)
rm -rf *.zip

mv data/sketches data/sketches_test
mv babies_test/images data/babies_test
mv sunglasses_test/images data/sunglasses_test
mv data/BitmojiDataset/images data/bitmoji_test
rmdir babies_test sunglasses_test
rm -r data/BitmojiDataset

# Resize images to output resolution
python resize.py data/babies_test &
python resize.py data/sketches_test --extension jpg &
python resize.py data/bitmoji_test --extension jpg & 
python resize.py data/sunglasses_test &

wait < <(jobs -p)

for dataset in {babies,bitmoji,sketches,sunglasses}
#for dataset in {babies,sketches,sunglasses}
do
	mkdir -p data/$dataset
	samples=$(ls data/"$dataset"_test/*)
	samples=$(shuf -e $samples)
	samples=( $samples )
	samples=${samples[@]:0:10}
	mv $samples data/$dataset
done
