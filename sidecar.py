import os
import sys

import torch
import json
import argparse
import torchvision as tv

import models
from utils import make_coord
from test import batched_predict_with_feat

def eprint(*args, **kwargs): print(*args, file=sys.stderr, **kwargs)

def arguments():
  a = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  a.add_argument(
    "--model",
    type=str,
    default="save/_train_edsr-liif-small/epoch-best.pth", help="Which model to use for super resolution"
  )
  a.add_argument("--total-size", type=int, default=30, help="Total number of expected channels")
  return a.parse_args()

N = 5
def super_resolve_video(model, chunk:int, max_latent:int, height=240, width=360):
  # read entire chunk in
  # TODO need to read more than the first frame and stick them all together
  frames = None
  for i in range(max_latent):
    file_name = f"vid.{chunk:03}.{i:03}.mp4"
    if not os.path.exists(file_name): break
    new_frames, _, info = tv.io.read_video(file_name)
    frames = new_frames if frames is None else torch.cat([frames, new_frames], dim=-1)

  if frames is None:
    eprint(f"No frames exist for video {chunk}")
    return
  frames = (2*(frames/255))-1
  frames = torch.cat([
    frames,
    torch.zeros(*frames.shape[:-1], 60-frames.shape[-1])
  ], dim=-1)
  frames = frames[..., :30]

  fps = info["video_fps"]
  # TODO possibly just make this on the GPU without passing it around??
  coord = make_coord(height//N, width//N).cuda()
  cell = torch.ones_like(coord)
  cell[:, 0] *= 2 / height
  cell[:, 1] *= 2 / width
  B = 2
  # batch along time dimension to fit in memory
  preds = []
  for subframes in frames.split(B, dim=0):
    subframes = subframes.movedim(-1,1)
    pred = batched_predict_with_feat(
      model,
      subframes.cuda(),
      coord[None].expand(B, -1, -1),
      cell[None].expand(B, -1, -1),
      bsize=50_000,
    ).cpu()
    preds.append(pred)
  preds = torch.cat(preds, dim=0)
  # pred in range -1,1
  eprint(pred.shape)
  exit()
  return fps

def main():
  args = arguments()
  eprint("Started")
  save_state = torch.load(args.model)
  model = models.make(save_state["model"], load_sd=True).eval().cuda()
  #model = torch.jit.optimize_for_inference(torch.jit.script(model))
  # expect every line on stdin to be a separate json blob.
  for l in sys.stdin:
    super_res_args = json.loads(l)
    out_filename = super_resolve_video(model, super_res_args["chunk"], super_res_args["max_latent"])
    print(out_filename)

if __name__ == "__main__":
  with torch.no_grad():
    main()
