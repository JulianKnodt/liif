import torch
import torchvision as TV
import torchvision.transforms.functional as TVF
import argparse
import os
import models
from models.liif import LIIF
from pathlib import Path
from tqdm import tqdm

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
  a.add_argument(
    "--chunk-sec", type=float, default=2., help="If chunking video, number of seconds to chunk by"
  )

  args = a.parse_args()
  assert(os.path.exists(args.hdr_video)), "Video file does not exist"
  assert(os.path.exists(args.model)), "Model does not exist"
  return args

def main():
  args = arguments()
  device = args.device
  print(f"[info]: Reading video from {args.hdr_video}")

  curr_start = 0
  save_file = torch.load(args.model, map_location=device)
  model = models.make(save_file["model"], load_sd=True).to(device).eval()

  print("[info]: successfully loaded encoder")
  chunk_num = 0
  while True:
    frames, _, info = TV.io.read_video(
      args.hdr_video,
      start_pts=curr_start,
      end_pts=curr_start + args.chunk_sec,
      pts_unit="sec",
    )
    if frames.shape[0] == 1: break

    if args.resize_h is not None or args.resize_w is not None:
      h = args.resize_h or frames.shape[1]
      w = args.resize_w or frames.shape[2]
      frames = TVF.resize(frames.movedim(-1, 1), (h, w), antialias=True)

    all_feats = []
    for i, f in enumerate(tqdm(frames.split(args.encode_batch_size, dim=0))):
      f = f.to(device)
      assert(f.dtype == torch.uint8)
      # renormalize frame color to -1,1
      f = (f.float()/255) * 2 - 1

      feats = model.gen_feat(f)
      all_feats.append(feats)
    all_feats = (torch.cat(all_feats, dim=0)+1)/2
    all_feats = (all_feats * 255).to(torch.uint8)
    for f, feat in enumerate(all_feats.split(3, dim=1)):
      TV.io.write_video(
        f"encoded/{Path(args.hdr_video).stem}.chunk_{chunk_num:03}.feat_{f:03}.mp4",
        feat.movedim(1,-1),
        fps=info["video_fps"],
        # TODO can this be libx265?
        video_codec="libx264",
      )
    print(f"Finished chunk {chunk_num}")
    curr_start += args.chunk_sec
    chunk_num += 1
  return

if __name__ == "__main__":
  with torch.no_grad():
    main()
