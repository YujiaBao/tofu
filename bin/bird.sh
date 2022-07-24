cuda=0
src_dataset=bird_seabird
tar_dataset=bird_water

python src/main.py \
    --cuda ${cuda} \
    --src_dataset ${src_dataset} \
    --tar_dataset ${tar_dataset} \
    --transfer_ebd \
    --lr 0.0001 \
    --weight_decay 0.01 \
    --patience 15
