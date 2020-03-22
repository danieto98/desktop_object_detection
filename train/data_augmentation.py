import cv2
import os
import sys
from skimage import transform, util

directory = ""
result_dir = ""

# Get directory paths
if len(sys.argv) == 3:
    directory = sys.argv[1]
    result_dir = sys.argv[2]
else:
    exit("Usage: downscale_images.py input_dataset_path output_dataset_path")

# Iterate over all child directory names
child_dirs = next(os.walk(directory))[1]
for child_dir in child_dirs:
    # Iterate over all images in the given directory
    for filename in os.listdir(directory + "/" + child_dir):
        # If file is a jpg or png image
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # Get image from filename
            original = cv2.imread(directory + "/" + child_dir + "/" + filename, cv2.IMREAD_COLOR)
            orig_scikit = util.img_as_float(original)

            # Get vertically flipped image
            vflipped = cv2.flip(original, flipCode=0)

            # Get horizontally flipped image
            hflipped = cv2.flip(original, flipCode=0)

            # Get rotated images by 30 degrees
            rot30 = transform.rotate(orig_scikit, angle=30)

            # Add noise
            noisy = util.random_noise(orig_scikit)

            # Blur image
            blurry = cv2.GaussianBlur(orig_scikit, (5,5), 0)

            # Output all images
            cv2.imwrite(result_dir + "/" + child_dir + "/" + filename, original)
            cv2.imwrite(os.path.splitext(result_dir + "/" + child_dir + "/" + filename)[0] + "_vflipped.jpg", vflipped)
            cv2.imwrite(os.path.splitext(result_dir + "/" + child_dir + "/" + filename)[0] + "_hflipped.jpg", hflipped)
            cv2.imwrite(os.path.splitext(result_dir + "/" + child_dir + "/" + filename)[0] + "_rot30.jpg", util.img_as_ubyte(rot30))
            cv2.imwrite(os.path.splitext(result_dir + "/" + child_dir + "/" + filename)[0] + "_noisy.jpg", util.img_as_ubyte(noisy))
            cv2.imwrite(os.path.splitext(result_dir + "/" + child_dir + "/" + filename)[0] + "_blurry.jpg", util.img_as_ubyte(blurry))