train-div2k:
	python3 train_liif.py --config configs/train-div2k/train_edsr-baseline-liif.yaml
test-div2k-4:
	python3 test.py --config configs/test/test-div2k-4.yaml \
    --model save/_train_edsr-baseline-liif/epoch-best.pth
test-div2k-2:
	python3 test.py --config configs/test/test-div2k-2.yaml \
    --model save/_train_edsr-baseline-liif/epoch-best.pth
