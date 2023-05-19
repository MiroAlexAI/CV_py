import tensorflow as tf

# загрузка картинки
img = tf.keras.preprocessing.image.load_img('path/to/image.jpg')

# преобразование картинки в тензор
img_tensor = tf.keras.preprocessing.image.img_to_array(img)

# увеличение разрешения картинки
upscaled_img = tf.image.resize(img_tensor, size=(2 * img_tensor.shape[0], 2 * img_tensor.shape[1]))

# преобразование тензора в картинку
upscaled_img = tf.keras.preprocessing.image.array_to_img(upscaled_img)

# сохранение увеличенной картинки
upscaled_img.save('path/to/upscaled_image.jpg')