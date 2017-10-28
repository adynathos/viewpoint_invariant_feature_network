
python3 tfeat_training.py \
	--net_type "intensity" \
	--out_dir "../../out/../../out/models/arch_flat_int" \
	--dataset_dir "../../datasets/generated/arch/flat" \
	--batch_size 256 \
	--epoch_size 1280000 \

python3 tfeat_training.py \
	--net_type "intensity" \
	--out_dir "../../out/../../out/models/arch_unw_int" \
	--dataset_dir "../../datasets/generated/arch/unwarp_both" \
	--batch_size 256 \
	--epoch_size 1280000 \

python3 tfeat_training.py \
	--net_type "depth" \
	--out_dir "../../out/../../out/models/arch_depth" \
	--dataset_dir "../../datasets/generated/arch/flat" \
	--batch_size 256 \
	--epoch_size 1280000 \

python3 tfeat_training.py \
	--net_type "normals" \
	--out_dir "../../out/../../out/models/arch_normals" \
	--dataset_dir "../../datasets/generated/arch/flat" \
	--batch_size 256 \
	--epoch_size 1280000 \
