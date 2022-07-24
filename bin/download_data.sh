#!/bin/bash

# this includes beer, ask2me, and splits for bird
echo "Download beer, ask2me and splits for bird"
wget https://people.csail.mit.edu/yujia/files/tofu/datasets.zip
unzip datasets.zip
rm datasets.zip
echo "Done"


# download bird
echo "Download bird"
cd datasets/bird

wget https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz
tar -xvf waterbird_complete95_forest2water2.tar.gz
mv *.json waterbird_complete95_forest2water2/
rm waterbird_complete95_forest2water2.tar.gz

# celeba and mnist will be automatically downloaded at run time
