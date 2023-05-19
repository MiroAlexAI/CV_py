
import super_image
import PIL

import requests


#url = 'https://online.fotolab.ru/uploaded/user_id_58/2022_10_18_154739.jpg'
#pict=requests.get(url, stream=True)

pict=('IMG_1.JPG')

image = PIL.Image.open(pict)
model = super_image.EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=4)
inputs = super_image.ImageLoader.load_image(image)
preds = model(inputs)

super_image.ImageLoader.save_image(preds, './3scaled_4x'+pict)
super_image.ImageLoader.save_compare(inputs, preds, './3scaled_4x_compare'+pict)