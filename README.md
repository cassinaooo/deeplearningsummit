# Transfer learning with VGG16: cats vs dogs

Material for a transfer learning tutorial prepared for the Deep Learning Summit in 2018. Starting from a VGG16 pre-trained on ImageNet, the tutorial trains only a new "head" to tell cats from dogs. It then opens up the network to show what it has learned: filters, the patterns that maximize neurons, class activation maps and a t-SNE projection of the feature space.

The notebooks and their comments are in Portuguese.

![t-SNE projection of cat and dog photos](images/tsne-cats-dogs.jpg)

*Cat and dog photos placed by a t-SNE projection of the trained model's `fc1` vectors. The projection never sees a label, yet the cats gather at the top and the dogs at the bottom.*

## Tutorial outline

The main notebook is [`Transfer Learning.ipynb`](Transfer%20Learning.ipynb).

### 1. The cost of training from scratch

VGG16 ([Simonyan & Zisserman, 2014](https://arxiv.org/abs/1409.1556)) has 138,357,544 parameters. Training a single network took 3 weeks on 4 NVIDIA GTX Titan Black GPUs. The three dense layers at the end hold 89.36% of those parameters.

### 2. The convolutional network as a feature extractor

To show that the convolutional part already works as a general-purpose feature extractor, the tutorial takes 3 keyframes from a video, showing a book, a Raspberry Pi and a phone, and computes the output vector for each. It then measures, frame by frame, the Euclidean distance to those vectors. The network was never trained to recognize any of these objects, yet similar frames land close together ([script](video/vector_for_frames.py)).

[![Demo video on YouTube](https://img.youtube.com/vi/eDeG8ubsxjU/hqdefault.jpg)](https://www.youtube.com/watch?v=eDeG8ubsxjU)

*Click to watch the demo on YouTube. The video is also in [`video/demo_transfer_learning.mp4`](video/demo_transfer_learning.mp4).*

### 3. Transfer learning: swapping the network's head

The 1,000-class ImageNet classifier comes off, replaced by global average pooling, a 32-unit dense layer and a 2-class softmax. The convolutional layers stay frozen, so only the two new layers are trained. Training uses the [Cats vs Dogs](https://www.microsoft.com/en-us/download/details.aspx?id=54765) dataset, with data augmentation, RMSprop and **a single epoch**.

<p align="center"><img src="modifications.png" width="560" alt="Diagram of the 1000-class classifier being replaced by a 2-class one"></p>

| Notebook | Training images | Validation images | Validation accuracy |
|---|---|---|---|
| `Transfer Learning.ipynb` | 5,600 | 1,400 | 96.1% |
| `Transfer Learningv2.ipynb` | 23,529 | 1,240 | 95.9% |

### 4. Looking inside the network

**First-layer filters.** Applied directly to a photo, the `block1_conv1` filters respond to edges and colours, which is what the network learned on ImageNet.

![A dog photo and the output of four first-layer convolutional filters](images/conv1-filters-applied.jpg)

**Where the network looks.** The class activation map (`visualize_cam`, from [keras-vis](https://github.com/raghakot/keras-vis)) highlights the regions that weighed most in the "cat" prediction. In this image of several cats, the network focuses on their heads.

<p align="center"><img src="images/grad-cam-cats.png" width="400" alt="Class activation map over a photo of several cats"></p>

**The ideal "cat" and "dog".** An image is optimized by gradient ascent to maximize each output neuron. In this example, from a model with a 1,024-unit dense layer, fur textures, eyes and snouts appear.

<p align="center"><img src="images/cat_1024.png" width="260" alt="Image that maximizes the cat neuron"> <img src="images/dog_1024.png" width="260" alt="Image that maximizes the dog neuron"></p>

*Left: the image that maximizes the "cat" neuron. Right: the one that maximizes the "dog" neuron.*

### 5. Visualizing the feature space

The `fc1` vectors are reduced with SVD and then projected to 2 dimensions with t-SNE, and each image is plotted at its position. The figure at the top of this README was made this way by the `Generate Vector` and `TSNE` notebooks, using the model saved in `data/`. The full-resolution version (10800×7200) is in [`data/tsne.png`](data/tsne.png).

## Files

| File | Contents |
|---|---|
| `Transfer Learning.ipynb` | The full tutorial: VGG16, transfer learning, visualizations and t-SNE |
| `Transfer Learningv2.ipynb` | The same sequence, trained on the full dataset |
| `Infer + Class Maps.ipynb` | Inference with the saved model, activation maps, saliency and filters |
| `Generate Vector.ipynb`, `TSNE.ipynb` | Compute the `fc1` vectors for 5,000 images and the t-SNE projection with the photos |
| `split_files.ipynb` | Splits the dataset into training and validation folders |
| `descorrompe_imgs.bash` | Deletes dataset files that aren't valid JPEGs |
| `convolution_intro/` | An introduction to convolution: an edge-detection filter applied with NumPy/SciPy |
| `video/` | Script and keyframes for the vector-distance experiment, plus the final video |
| `data/` | The trained model (`.json` + `.h5`, with a 1,024-unit `fc1`), the saved vectors and the full-resolution t-SNE |

## Running it

The code dates from January 2018. It was written for Python 3.5, Keras 2.0 with the TensorFlow 1.x backend and [keras-vis](https://github.com/raghakot/keras-vis), plus scikit-learn, scikit-image, imageio and tqdm. It won't run on a current setup without changes, because the Keras and keras-vis APIs have changed since then. The data paths are also hard-coded (`/home/sage/...`) and need to be adjusted.

To reproduce it:

1. Download the [Cats vs Dogs](https://www.microsoft.com/en-us/download/details.aspx?id=54765) dataset.
2. Run `descorrompe_imgs.bash` inside the image folder to remove corrupted files.
3. Split training and validation with `split_files.ipynb`, into `train/{cat,dog}` and `val/{cat,dog}`.
4. Work through `Transfer Learning.ipynb`.

## License

GPL-3.0, see [LICENSE](LICENSE).
