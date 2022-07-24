cuda=0
#beer_0, beer_1, beer_2 correspond to look, aroma, palate respectively
src_dataset=beer_0
tar_dataset=beer_1

python src/main.py \
    --cuda ${cuda} \
    --src_dataset ${src_dataset} \
    --tar_dataset ${tar_dataset} \
    --transfer_ebd \
    --lr 0.001 \
    --patience 15
