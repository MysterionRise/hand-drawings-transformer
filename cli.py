import glob
import sys

from transformer.image import process

if __name__ == "__main__":
    pattern = f"resources/images/{sys.argv[1] if len(sys.argv) > 1 else '*'}.jpg"
    for file in glob.glob(pattern):
        process(file, "static/images")
