import sys
from PIL import Image

def rescale_image(image_path, x_percentage, y_percentage, in_path, out_path):
  path = f"{in_path}{image_path}"
  print(path)
  image = Image.open(path)

  new_width = int(image.width * (x_percentage / 100))
  new_height = int(image.height * (y_percentage / 100))

  resized_image = image.resize((new_width, new_height))

  image_title = image_path.split(".")[0]
  resized_image.save(f"{out_path}{image_title}_out_rescaled.bmp")

image_path = sys.argv[1]
x_percentage = float(sys.argv[2])
y_percentage = float(sys.argv[3])
in_path = sys.argv[4]
out_path = sys.argv[5]

rescale_image(image_path, x_percentage, y_percentage, in_path, out_path)
