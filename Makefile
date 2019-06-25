up:
	git add -A
	git commit -m "AUTO: Small Fix"
	git push

run:
	python3 train.py

run_gpu:
	python3 train.py --gpu

sync:
	rsync -arvu --exclude=logs_v2/ -e ssh . naturalreaders:spair_pytorch

tb:
	tensorboard --logdir logs_v2/ --host 0.0.0.0 --port 8081