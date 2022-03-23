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

encode-sample:
	python3 encode_video.py --model save/_train_edsr-baseline-liif/epoch-best.pth \
    --hdr /home/kadmin/nerf_atlas/data/shoichi_chris.mp4 --resize-w 480 --resize-h 270
