clean:
	-@rm outputs/*.png

train-div2k:
	python3 train_liif.py --config configs/train-div2k/train_edsr-baseline-liif.yaml

# lower memory usage?
test-div2k-4: clean
	python3 test.py --config configs/test/test-div2k-4.yaml \
    --model save/_train_edsr-baseline-liif/epoch-best.pth

# higher memory usage?
test-div2k-2: clean
	python3 test.py --config configs/test/test-div2k-2.yaml \
    --model save/_train_edsr-baseline-liif/epoch-best.pth

