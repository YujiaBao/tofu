cuda=0
src_dataset=ask2me_penetrance
tar_dataset=ask2me_incidence

python src/main.py \
    --cuda ${cuda} \
    --src_dataset ${src_dataset} \
    --tar_dataset ${tar_dataset} \
    --transfer_ebd \
    --lr 0.001 \
    --patience 15
