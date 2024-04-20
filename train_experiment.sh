#! /bin/bash

GPU=$1
SELECTION=$2
RUN=$3

if [ $SELECTION = "tiny" ]
then
    # tinyimagenet
    dataset=tinyimagenet
    model=resnet188
    wd=0.0001
    bs=64
    gamma1=10000
elif [ $SELECTION = "cifar" ]
then
    # cifar
    dataset=cifar
    model=resnet188
    wd=0.0005
    bs=32
    gamma1=10000
elif [ $SELECTION = "mnist" ]
then
    # mnist
    model=simple
    dataset=mnist
    wd=0.0001
    bs=32
    gamma1=10000
else
    echo "invalid selection $SELECTION"
    exit -1
fi

ci=.75

# lisa
for i in {1..5}
do
    comment="_${RUN}_run-${i}_gamma1-${gamma1}_ci-${ci}"
    CUDA_VISIBLE_DEVICES=$GPU python main.py --dataset=$dataset --batch-size=$bs \
                                            --optim="lisa-vm" \
                                            --steps=50000 --beta1=0.99 --beta2=0.9 --beta3=0.999 --alpha=0.00001 \
                                            --ls_ci=${ci} --gamma1=$gamma1\
                                            --weight-decay=$wd \
                                            --log-interval=10 --test-interval=1000 \
                                            --model=$model \
                                            --comment=$comment

    # adam
    comment="_run-${RUN}"
    CUDA_VISIBLE_DEVICES=$GPU python main.py --dataset=$dataset --batch-size=$bs \
                                            --optim="adam" \
                                            --steps=50000 --acc=0.9 --beta1=0.9 --beta2=0.999 --alpha=0.00025 \
                                            --weight-decay=$wd \
                                            --log-interval=100 --test-interval=1000 \
                                            --model=$model \
                                            --comment=$comment

    # adabelief
    comment="_run-${RUN}"
    CUDA_VISIBLE_DEVICES=$GPU python main.py --dataset=$dataset --batch-size=$bs \
                                            --optim="adabelief" \
                                            --steps=50000 --alpha=0.00025 --acc=0.9 --beta1=.9 --beta2=0.999\
                                            --weight-decay=$wd \
                                            --log-interval=100 --test-interval=1000 \
                                            --model=$model \
                                            --comment=$comment

done

