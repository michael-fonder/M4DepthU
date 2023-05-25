#!/bin/bash

dataset=$1;

if [ ! -z "$2" ]
then
    savepath=$2
fi

case "$dataset" in

    "midair")
        if [ -z "$2" ]
        then
            savepath="pretrained_weights/midair_3"
        fi
        data="data/midair/test_data"
        ;;

    "midair-cloudy")
        if [ -z "$2" ]
        then
            savepath="pretrained_weights/midair_3"
        fi
        data="data/midair-eval/midair-cloudy"
        dataset="midair"
        ;;

    "midair-fall")
        if [ -z "$2" ]
        then
            savepath="pretrained_weights/midair_3"
        fi
        data="data/midair-eval/midair-fall"
        dataset="midair"
        ;;

    "midair-foggy")
        if [ -z "$2" ]
        then
            savepath="pretrained_weights/midair_3"
        fi
        data="data/midair-eval/midair-foggy"
        dataset="midair"
        ;;

    "midair-spring")
        if [ -z "$2" ]
        then
            savepath="pretrained_weights/midair_3"
        fi
        data="data/midair-eval/midair-spring"
        dataset="midair"
        ;;

    "midair-winter")
        if [ -z "$2" ]
        then
            savepath="pretrained_weights/midair_3"
        fi
        data="data/midair-eval/midair-winter"
        dataset="midair"
        ;;

    "midair-sunny")
        if [ -z "$2" ]
        then
            savepath="pretrained_weights/midair_3"
        fi
        data="data/midair-eval/midair-sunny"
        dataset="midair"
        ;;

    "midair-sunset")
        if [ -z "$2" ]
        then
            savepath="pretrained_weights/midair_3"
        fi
        data="data/midair-eval/midair-sunset"
        dataset="midair"
        ;;

    "kitti")
        if [ -z "$2" ]
        then
            savepath="pretrained_weights/midair_4"
        fi
        data="data/kitti-raw-filtered/test_data"
        dataset="kitti-raw"
        ;;

    "rmvd")
        if [ -z "$2" ]
        then
            savepath="pretrained_weights/midair_4"
        fi
        data="data/kitti-raw-rmvd/test_data"
        dataset="kitti-raw"
        ;;

    "tartanair-gascola")
        if [ -z "$2" ]
        then
            savepath="pretrained_weights/midair_1"
        fi
        data="data/tartanair/unstructured/test_data/gascola"
        dataset="tartanair"
        ;;

    "tartanair-winter")
        if [ -z "$2" ]
        then
            savepath="pretrained_weights/midair_1"
        fi
        data="data/tartanair/unstructured/test_data/seasonsforest_winter"
        dataset="tartanair"
        ;;
    
    "tartanair-neighborhood")
        if [ -z "$2" ]
        then
            savepath="pretrained_weights/midair_1"
        fi
        data="data/tartanair/urban/test_data/neighborhood"
        dataset="tartanair"
        ;;

    "tartanair-oldtown")
        if [ -z "$2" ]
        then
            savepath="pretrained_weights/midair_1"
        fi
        data="data/tartanair/urban/test_data/oldtown"
        dataset="tartanair"
        ;;

    *)
        echo "ERROR: Wrong dataset argument supplied"
        ;;
esac

python main.py --mode=eval --dataset="$dataset" --arch_depth=6 --ckpt_dir="$savepath" --records="$data" $3
