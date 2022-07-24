cuda=0
src_dataset=celeba:Eyeglasses:Young
tar_dataset=celeba:Blond_Hair:Young

python src/main.py \
    --batch_size 50 \
    --cuda ${cuda} \
    --src_dataset ${src_dataset} \
    --tar_dataset ${tar_dataset} \
    --transfer_ebd \
    --lr 0.0001 \
    --weight_decay 0.01 \
    --patience 15
