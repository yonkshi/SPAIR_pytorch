up:
	git add -A
	git commit -m "AUTO: Small Fix"
	git push

run:
	python3 train.py

run_gpu:
	python3 train.py --gpu --conv_spair

sync:
	rsync -arvu --exclude=logs_v2/ --exclude=logs/ --exclude=spair/data/* --exclude=data/* -e ssh . naturalreaders:spair_pytorch

tb:
	tensorboard --logdir logs/ --host 0.0.0.0 --port 8081


overnight:
	python3 train.py --gpu || true
	python3 train.py --gpu --no_z_prior || true
	python3 train.py --gpu --uniform_z_prior || true
	python3 train.py --gpu --conv_spair || true
	python3 train.py --gpu --no_z_prior --conv_spair || true
	python3 train.py --gpu --uniform_z_prior --conv_spair || true
