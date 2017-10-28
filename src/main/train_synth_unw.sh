
python3 tfeat_training.py \
	--net_type "intensity" \
	--dataset_dir "../../datasets/generated/synthetic/unwarp_both" \
	--out_dir "../../out/models/synth_unw" \
	--batch_size 256 \
	--epoch_size 1280000 \
	--epochs 40 \
