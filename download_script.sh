#!/bin/bash

export KAGGLE_USERNAME="adnanharundoan" KAGGLE_KEY="51b9a167d974acb6bf619f1bd9e92ebc"

python -c "import gdown; url = 'https://drive.google.com/u/0/uc?id=1PQutd-JboOCOZqmd95XWxWrO8gGEvRcO&export=download'; output = '550000.pt'; gdown.download(url, output, quiet=True)"

python -c "import gdown; url = 'https://drive.google.com/u/0/uc?id=1Uu5y_y8Rjxbj2VEzvT3aBHyn4pltFgyX&export=download'; output = 'sunglasses_real_test.zip'; gdown.download(url, output, quiet=True)"

python -c "import gdown; url = 'https://drive.google.com/u/0/uc?id=1JmjKBq_wylJmpCQ2OWNMy211NFJhHHID'; output = 'babies_real_test.zip'; gdown.download(url, output, quiet=True)"

kaggle datasets download -d mostafamozafari/bitmoji-faces &
wget --quiet http://mmlab.ie.cuhk.edu.hk/archive/sketchdatabase/CUHK/training_88/Cropped_Images/CUHK_training_cropped_sketches.zip &

wait < <(jobs -p)

mkdir -p data
unzip -q babies_real_test.zip -d babies_test &
unzip -q bitmoji-faces.zip -d data &
unzip -q sunglasses_real_test.zip -d sunglasses_test &
unzip -q CUHK_training_cropped_sketches.zip -d data &

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
do
	mkdir -p data/$dataset
	samples=$(ls data/"$dataset"_test/*)
	samples=$(shuf -e $samples)
	samples=( $samples )
	samples=${samples[@]:0:10}
	mv $samples data/$dataset
done
