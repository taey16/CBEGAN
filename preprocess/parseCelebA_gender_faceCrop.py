import os
import sys
import re
import cv2

cascPath = './haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascPath)
#celebRawImgRoot = '/home1/taey16/storage/CelebA/CelebA/Img/img_align_celeba/%s'
celebRawImgRoot = '/path/to/raw/image/dir/img_align_celeba/%s'
outputSize = 128

#entries = [entry.strip() for entry in open('../Anno/list_attr_celeba.data.txt', 'r')]
entries = [entry.strip() for entry in open('/path/to/Anno/file/list_attr_celeba.data.txt', 'r')]

#import pdb; pdb.set_trace()
for entry in entries:
  entry = re.split('\s+', entry) # split
  fname = entry[0] # get filename
  fname_noExt = int(fname[:-4]) # filename \wo extention
  gender_flag = int(entry[21]) # gneder flag
  imgPath = celebRawImgRoot % fname # get input raw image path
  image = cv2.imread(imgPath)
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  faces = faceCascade.detectMultiScale(gray, 5, 5)
  if len(faces) == 0:
    print('ERROR in %s' % imgPath)
    sys.stdout.flush()
    pass
  else:
    # crop face and then resize
    x,y,w,h = faces[0]
    cropped = image[y:y+w, x:x+w, :]
    outputImg = cv2.resize(cropped, (outputSize, outputSize))
    # train/val/test split
    if fname_noExt < 162771: #trn
      if gender_flag == 1:
        #outF = '/home1/taey16/storage/CelebA/CelebA/gender_facecrop/train/imgA/%s.png' % fname[:-4]
        outF = '/path/to/gender_facecrop/train/imgA/%s.png' % fname[:-4]
        cv2.imwrite(outF, outputImg)
      else:
        #outF = '/home1/taey16/storage/CelebA/CelebA/gender_facecrop/train/imgB/%s.png' % fname[:-4]
        outF = '/path/to/gender_facecrop/train/imgB/%s.png' % fname[:-4]
        cv2.imwrite(outF, outputImg)
    elif fname_noExt >= 162771 and fname_noExt < 182638:
      if gender_flag == 1:
        #outF = '/home1/taey16/storage/CelebA/CelebA/gender_facecrop/val/imgA/%s.png' % fname[:-4]
        outF = '/path/to/gender_facecrop/val/imgA/%s.png' % fname[:-4]
        cv2.imwrite(outF, outputImg)
      else:
        #outF = '/home1/taey16/storage/CelebA/CelebA/gender_facecrop/val/imgB/%s.png' % fname[:-4]
        outF = '/path/to/gender_facecrop/val/imgB/%s.png' % fname[:-4]
        cv2.imwrite(outF, outputImg)
    else:
      if gender_flag == 1:
        #outF = '/home1/taey16/storage/CelebA/CelebA/gender_facecrop/test/imgA/%s.png' % fname[:-4]
        outF = '/path/to/gender_facecrop/test/imgA/%s.png' % fname[:-4]
        cv2.imwrite(outF, outputImg)
      else:
        #outF = '/home1/taey16/storage/CelebA/CelebA/gender_facecrop/test/imgB/%s.png' % fname[:-4]
        outF = '/path/to/gender_facecrop/test/imgB/%s.png' % fname[:-4]
        cv2.imwrite(outF, outputImg)
    print('End of %s, gender: %d' % (outF, gender_flag))
