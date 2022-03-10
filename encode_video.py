import torc
import torchvision as TV
import argparse
import os

def arguments():
  a = argparse.ArgumentParser()
  a.add_argument("--hdr-video", required=True, help="High resolution video to encode", type=str)
  a.add_argument("--model", required=True, help="Model to use for encoding the input", type=str)
  a.add_argument("--device", type=str, default="cuda", help="Device to run encoding on")

  args = a.parse_args()
  assert(os.path.exists(args.hdr_video)), "Video file does not exist"
  assert(os.path.exists(args.model)), "Model does not exist"
  return args

def main():
  args = arguments()
  device=args.device
  model = torch.load(args.model, map_location=device)
  frames = TV.read_video(args.hdr_video)
  print(f"[info]: {frames.shape[0]} frames in video.")
  encoder = model.encoder
  manifest = []
  for f in frames.split(1, dim=1):
    feat = encoder(f)
    print(feat.shape)
    # TODO write this to a file, or encode it somehow?
    # Idea: ensure that the feat size is divisible by 3.
    # Then we can use a normal encoder in order to ensure that we are properly compressing the
    # data?

    # TODO we are also interested in the error of certain regions, which can be used later to
    # inform which parts need more info
    frame_summary = {
      "error_file": 0,
      "feat_file": 0,
      "hdr_file": 0,
    }
    manifest.append(frame_summary)
    exit()
  return
