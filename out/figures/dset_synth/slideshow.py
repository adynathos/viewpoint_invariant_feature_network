import numpy as np
import cv2
import glob


imgs = [cv2.imread(fp) for fp in sorted(glob.glob('src/*.jpg'))]

img_merged = np.concatenate(imgs, axis=1)

cv2.imwrite('merged.jpg', img_merged)

img_merged_sml = cv2.resize(img_merged, None, fx=1/4, fy=1/4)

img_merged_sml[:, ::256, :] = 255

cv2.imwrite('merged_sml.jpg', img_merged_sml)

