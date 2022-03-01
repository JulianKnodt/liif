train-div2k:
	python3 train_liif.py --config configs/train-div2k/train_edsr-baseline-liif.yaml
test-div2k:
	test.py --config configs/test/test-div2k-4.yaml \
    --model save/_train_edsr-baseline-liif/epoch-best.pth
