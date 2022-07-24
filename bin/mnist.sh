cuda=0
src_dataset=MNIST_02468_01234_5
tar_dataset=MNIST_13579_01234_5

python src/main.py \
    --cuda ${cuda} \
    --src_dataset ${src_dataset} \
    --tar_dataset ${tar_dataset} \
    --transfer_ebd \
    --lr 0.001 \
    --weight_decay 0.01 \
    --patience 5
