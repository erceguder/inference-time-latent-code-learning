from PIL import Image
from tqdm import tqdm
import glob
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("directory", help="directory of images to be resized")
    parser.add_argument("--resolution", type=int, default=256)

    args = parser.parse_args()

    files = glob.glob(f"{args.directory}/*.png")

    for file in tqdm(files):
        im = Image.open(file)
        im = im.resize((args.resolution, args.resolution))
        im.save(file)