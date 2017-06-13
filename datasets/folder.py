import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np

IMG_EXTENSIONS = [
  '.jpg', '.JPG', '.jpeg', '.JPEG',
  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
  return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def find_classes(dir):
  classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
  classes.sort()
  class_to_idx = {classes[i]: i for i in range(len(classes))}
  return classes, class_to_idx


def make_dataset(dir, class_to_idx):
  images = []
  for target in os.listdir(dir):
    d = os.path.join(dir, target)
    if not os.path.isdir(d):
      continue

    for root, _, fnames in sorted(os.walk(d)):
      for fname in fnames:
        if is_image_file(fname):
          path = os.path.join(root, fname)
          item = (path, class_to_idx[target])
          images.append(item)

  return images


def pil_loader(path):
  # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
  """
  with open(path, 'rb') as f:
    with Image.open(f) as img:
      return img.convert('RGB')
  """
  f = open(path, 'rb')
  img = Image.open(f)
  img = img.convert('RGB')
  f.close()
  return img


def accimage_loader(path):
  import accimage
  try:
    return accimage.Image(path)
  except IOError:
    # Potentially a decoding problem, fall back to PIL.Image
    return pil_loader(path)


def default_loader(path):
  """
  from torchvision import get_image_backend
  if get_image_backend() == 'accimage':
    return accimage_loader(path)
  else:
    return pil_loader(path)
  """
  return pil_loader(path)


class ImageFolder(data.Dataset):

  def __init__(self, root, transform=None, target_transform=None,
               loader=default_loader, seed=None):
    classes, class_to_idx = find_classes(root)
    imgs = make_dataset(root, class_to_idx)
    if len(imgs) == 0:
      raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                         "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

    self.root = root
    self.imgs = imgs
    self.classes = classes
    self.class_to_idx = class_to_idx
    self.transform = transform
    self.target_transform = target_transform
    self.loader = loader

    if seed is not None:
      np.random.seed(seed)

    self.len_img = len(self.imgs)

  def __getitem__(self, _):
    #path, target = self.imgs[index]
    index = np.random.randint(self.__len__(), size=1)[0]
    path, target = self.imgs[index]
    img = self.loader(path)
    if self.transform is not None:
      img = self.transform(img)
    if self.target_transform is not None:
      target = self.target_transform(target)

    return img, target

  def __len__(self):
    return len(self.imgs)
