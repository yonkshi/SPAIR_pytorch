up:
	git add -A
	git commit -m "AUTO: Small Fix"
	git push

run:
	python3 train.py

run_gpu:
	python3 train.py --gpu

sync:
	rsync -arvu --exclude=logs_v2/ --exclude=logs/ --exclude=spair/data/* --exclude=data/* -e ssh . naturalreaders:spair_pytorch

tb:
	tensorboard --logdir logs/ --host 0.0.0.0 --port 8081


overnight:
	python3 train.py --gpu || true
	python3 train.py --gpu || true
	python3 train.py --gpu --z_pres no_prior  || true
	python3 train.py --gpu --z_pres uniform_prior || true
	python3 train.py --gpu --original_spair  || true
	python3 train.py --gpu --original_spair  --z_pres no_prior || true
	python3 train.py --gpu --original_spair  --z_pres uniform_prior || true

test_new_features:
	python3 train.py --gpu --use_z_where_decoder || true
	python3 train.py --gpu --use_uber_trick || true
	python3 train.py --gpu --use_conv_z_attr || true
	python3 train.py --gpu --z_pres none || true
	python3 train.py --gpu --z_pres uniform || true
