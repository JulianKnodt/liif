import torch
import torchvision as TV
import torchvision.transforms.functional as TVF
import argparse
import os
import models
from models.liif import LIIF
from pathlib import Path

def arguments():
  a = argparse.ArgumentParser()
  a.add_argument("--hdr-video", required=True, help="High resolution video to encode", type=str)
  a.add_argument("--model", required=True, help="Model to use for encoding the input", type=str)
  a.add_argument("--device", type=str, default="cuda", help="Device to run encoding on")
  a.add_argument(
    "--resize-w", type=int, default=None, help="Resize video to this width before encoding"
  )
  a.add_argument(
    "--resize-h", type=int, default=None, help="Resize video to this height before encoding"
  )
  a.add_argument(
    "--encode-batch-size", type=int, default=3, help="Batch size to use while encoding video"
  )

  args = a.parse_args()
  assert(os.path.exists(args.hdr_video)), "Video file does not exist"
  assert(os.path.exists(args.model)), "Model does not exist"
  return args

def main():
  args = arguments()
  device = args.device
  print(f"[info]: Reading video from {args.hdr_video}")

  frames, _, _ = TV.io.read_video(args.hdr_video, end_pts=0.5, pts_unit="sec")

  print(f"[info]: {frames.shape[:-1]} frames in video")
  if args.resize_h is not None or args.resize_w is not None:
    h = args.resize_h or frames.shape[1]
    w = args.resize_w or frames.shape[2]
    frames = TVF.resize(frames.movedim(-1, 1), (h, w), antialias=True)
  save_file = torch.load(args.model, map_location=device)
  model = models.make(save_file["model"], load_sd=True).to(device).eval()

  print("[info]: successfully loaded encoder")
  for i, f in enumerate(frames.split(args.encode_batch_size, dim=0)):
    f = f.to(device)
    assert(f.dtype == torch.uint8)
    # renormalize frame color to -1,1
    f = (f.float()/255) * 2 - 1

    feats = model.gen_feat(f)
    for j, feat in enumerate(feats.split(1, dim=0)):
      f = i * args.encode_batch_size + j
      for c, chan in enumerate(feat.squeeze(0).split(3, dim=0)):
        TV.utils.save_image(chan, f"encodings/{Path(args.hdr_video).stem}.f{f}.c{c}.jpg")
  return

if __name__ == "__main__":
  with torch.no_grad():
    main()
