from keras.models import load_model
from keras.models import model_from_json
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import matplotlib.pyplot as plt

model_location = '/home/sage/ml_playground/catsvsdogs/trained/keras/vgg16_cat_dog_retrained.json'
weights_location = '/home/sage/ml_playground/catsvsdogs/trained/keras/vgg16_cat_dog_retrained.h5'

with open(model_location, 'r') as myfile:
    model_json = myfile.read()

model = model_from_json(model_json)
model.load_weights(weights_location)

headless_model = Model(inputs=model.input, outputs=model.get_layer('global_average_pooling2d_1').output)

frames_dir = '/home/sage/ml_playground/catsvsdogs/keras-fine-tune-cat-dog-vgg16/video/frames'
keyframes_dir = '/home/sage/ml_playground/catsvsdogs/keras-fine-tune-cat-dog-vgg16/video/keyframes'
save_dir = '/home/sage/ml_playground/catsvsdogs/keras-fine-tune-cat-dog-vgg16/video/newframes'


img_width, img_height = 224, 224

frames = ImageDataGenerator().flow_from_directory(directory=frames_dir,
                                                              target_size=[img_width, img_height],
                                                              batch_size=1,
                                                              shuffle=False,
                                                              class_mode=None)

keyframes = ImageDataGenerator().flow_from_directory(directory=keyframes_dir,
                                                              target_size=[img_width, img_height],
                                                              batch_size=1,
                                                              shuffle=False,
                                                              class_mode=None)

livro = (255. - next(keyframes))
telefone = (255. - next(keyframes))
raspberry = (255. - next(keyframes))

livro_vector = headless_model.predict(livro)[0]
telefone_vector = headless_model.predict(telefone)[0]
raspberry_vector = headless_model.predict(raspberry)[0]

print('livro_vector_shape', livro_vector.shape)
print('telefone_vector_shape', telefone_vector.shape)
print('raspberry_vector_shape', raspberry_vector.shape)


from tqdm import tqdm

n_images = 1265

for i in tqdm(range(n_images)):

    img = (255. - next(frames))

    img_vector = headless_model.predict(img)[0]

    ax = plt.subplot(111)

    ax.set_frame_on(False)
    ax.set_xticks([])
    ax.set_yticks([])

    d_livro = np.linalg.norm(img_vector - livro_vector)
    d_tel = np.linalg.norm(img_vector - telefone_vector)
    d_rasp = np.linalg.norm(img_vector - raspberry_vector)

    mais_proximo = 'telefone'

    if(d_livro < d_rasp and d_livro < d_tel):
        mais_proximo = 'livro'
    elif(d_rasp < d_tel):
        mais_proximo = 'raspberry'

    ax.annotate('distancia livro= {:2.2f}'.format(d_livro), xy=(0,10), color='white')
    ax.annotate('distancia telefone = {:2.2f}'.format(d_tel), xy=(0,20), color='white')
    ax.annotate('distancia raspberry = {:2.2f}'.format(d_rasp), xy=(0,30), color='white')
    ax.annotate('menor distancia = ' + mais_proximo, xy=(0,40), color='white')    

    plt.imshow(img[0])

    file_name = frames.filenames[i].split('/')[-1]

    plt.savefig('newframes/{}'.format(file_name))

    plt.clf()
    plt.cla()
