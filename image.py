
import matplotlib.pyplot as plt
import matplotlib.image as mimg
import numpy as np
import PIL as pil

img = mimg.imread('BV5A2969-tech-demo.jpg')

imgplot = plt.imshow(img)

lumimg = img[:,:,0]

imgplot = plt.imshow(lumimg)

imgplot = plt.imshow(lumimg)
plt.colorbar()

# doesn't work because img/lumimg is 1-256 and demo was floating point??
plt.hist(lumimg.ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')

# use flatten and appropriate range

plt.hist(lumimg.flatten(), bins=256, range=(1,256))

# increase contrast and show
imgplot = plt.imshow(lumimg, clim=(0, 100))

# use PIL to resize...
small_img = pil.Image.open('BV5A2969-tech-demo.jpg')
small_img.thumbnail((128, 128), pil.Image.ANTIALIAS)  # resizes image in-place
imgplot = plt.imshow(small_img)

fimage = pil.Image.fromarray(lumimg)
fimage.thumbnail((28,28), pil.Image.ANTIALIAS)
imgplot = plt.imshow(fimage)

plt.hist(np.asarray(fimage).flatten(), bins=256, range=(1,256))

detect_img = (np.expand_dims(np.asarray(fimage),0))
print(detect_img.shape)

predictions_detect = probability_model.predict(detect_img)

print(predictions_detect)

plot_value_array(1, predictions_detect[0], test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)

#Extra stuff

oa = np.asarray(fimage)
cont_img = np.clip(oa, 10, 70)
imgplot = plt.imshow(cont_img)
plt.colorbar()

detect_img = (np.expand_dims(np.asarray(oa),0))
print(detect_img.shape)

predictions_detect = probability_model.predict(detect_img)

print(predictions_detect)

plot_value_array(1, predictions_detect[0], test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)