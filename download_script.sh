#!/bin/bash

wget https://user.ceng.metu.edu.tr/~adhd/courses/550000.pt
wget https://user.ceng.metu.edu.tr/~adhd/courses/babies_images.zip
unzip babies_images.zip -d babies

python -c "import gdown; url = 'https://drive.google.com/u/0/uc?id=1JmjKBq_wylJmpCQ2OWNMy211NFJhHHID'; output = 'babies_real_test.zip'; gdown.download(url, output, quiet=False)"
unzip babies_real_test.zip -d babies_test
