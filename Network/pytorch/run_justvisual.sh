#!/bin/bash
in=$1

name=full
command="CUDA_VISIBLE_DEVICES=0
		python3 ./main.py 
		--name $name 
		--weight 64.0
		--batch_size 8
		--n_threads 4 
		--lr 0.001
		--train_list 	../../Assets/training-data/filtered_trainset.json
		--val_list 		../../Assets/training-data/filtered_valset.json
	  --visual_list ../../Assets/training-data/generated_testset.json
    --just_visualize 1
		--mask_neg 1
		--with_scale 1
		--with_match 1
    --max_iteration 250000
		--output /data/scan2cad_output/
		--interval_eval 50
		--n_samples_eval 1024"

#--output /mnt/raid/armen/output/suncg/

echo "***************"
echo NAME: $name
echo $command
echo "***************"
echo ""

eval $command

