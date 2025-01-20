# Siamese-Neural-Network


## Siamese Neural Network for Image Comparison

This project implements a Siamese Neural Network using TensorFlow/Keras for comparing images and detecting similarities and differences between them. The network utilizes VGG16 as the backbone architecture and implements contrastive loss for training.

## Features

- Pre-trained VGG16 backbone for feature extraction
- Contrastive loss implementation for similarity learning
- Cosine similarity calculation for image comparison
- Difference visualization using heatmaps
- Support for custom input image sizes
- GPU-accelerated inference (when available)

## Requirements

```
tensorflow
numpy
matplotlib
opencv-python
```

## Architecture

The Siamese network consists of:
1. Two identical VGG16 networks (pre-trained on ImageNet)
2. Custom top layers for embedding generation
3. Contrastive loss function for training
4. Similarity computation using cosine similarity

## Installation

1. Clone this repository
2. Install the required packages:
```bash
pip install tensorflow numpy matplotlib opencv-python
```

## Usage

### Basic Usage

```python
from siamese_network import compare_images

# Compare two images
image_path1 = "path/to/first/image.jpg"
image_path2 = "path/to/second/image.jpg"

similarity_score, differences, visualization = compare_images(image_path1, image_path2)
```

### Model Creation

```python
from siamese_network import create_siamese_model

# Create a model with custom input shape
input_shape = (224, 224, 3)
model = create_siamese_model(input_shape)
```

## Model Details

The network processes images through the following steps:

1. Image Preprocessing:
   - Resizing to 224x224 pixels
   - VGG16 preprocessing
   
2. Feature Extraction:
   - VGG16 convolutional layers
   - Flattening
   - Dense layer with 128 units (ReLU activation)
   - Final embedding layer (128 dimensions)

3. Similarity Computation:
   - Cosine similarity between embeddings
   - Euclidean distance calculation
   - Heatmap generation for visualization

## Loss Function

The model uses contrastive loss defined as:

```python
def contrastive_loss(y_true, y_pred):
    margin = 1.0
    square = tf.square(y_pred)
    margin_square = tf.square(tf.maximum(margin - y_pred, 0))
    return tf.reduce_mean(y_true * square + (1 - y_true) * margin_square)
```

## Output

The comparison function returns:
- Similarity score (-1 to 1, higher means more similar)
- Numerical difference measure
- Visualization of differences as a heatmap

## Example Output

```python
Similarity score: -0.052438866
Differences: 203.6052
```

## Contributing

Feel free to open issues or submit pull requests for any improvements or bug fixes.

## License

[Add your chosen license here]

## Citation

If you use this code in your research, please cite:

[Add citation information if applicable]
