## CIFAR-10 Image Classification using ResNet50
This project implements image classification on the CIFAR-10 dataset using ResNet50 architecture and TensorFlow/Keras.

## Project Overview
The project uses the CIFAR-10 dataset, which consists of 60,000 32x32 color images in 10 different classes. The goal is to correctly classify these images using deep learning.

### Dataset Classes
- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

## Technical Implementation
### Dependencies
- tensorflow
- keras
- numpy
- pandas
- PIL
- matplotlib
- scikit-learn
- opencv-python
- py7zr
- opendatasets

## Project Structure

1.Data Loading and Preprocessing
- Download CIFAR-10 dataset using opendatasets
- Extract images using py7zr
- Convert images to numpy arrays
- Normalize pixel values (0-255 to 0-1)
- Split data into training and testing sets (80-20 split)

2.Model Architecture
- Base: ResNet50 pretrained on ImageNet
- Additional layers:
  - Multiple UpSampling2D layers
  - Flatten layer
  - Batch Normalization layers
  - Dense layers with ReLU activation
  - Dropout layers (0.5)
  - Final Dense layer with softmax activation (10 classes)


3.Training
- Optimizer: RMSprop with learning rate 2e-5
- Loss: Sparse Categorical Crossentropy
- Metrics: Accuracy
- Validation split: 10%
- Epochs: 10

## Results
- The model's performance is visualized through:
  - Training vs Validation Loss curves
  - Training vs Validation Accuracy curves

## Usage
1.Install required packages:
```python
pip install tensorflow keras numpy pandas pillow matplotlib scikit-learn opencv-python py7zr opendatasets
```
2.Run the notebook:
  - The notebook will download and process the CIFAR-10 dataset
  - Train the ResNet50 model
  - Display performance metrics

## Model Benefits
- Uses transfer learning with pretrained ResNet50
- Implements data augmentation
- Includes dropout for regularization
- Uses batch normalization for better training stability
- Implements modern deep learning best practices

## Future Improvements
- Implement data augmentation
- Try different architectures (VGG, Inception)
- Experiment with different optimizers
- Add model checkpointing
- Implement early stopping

## License
This project is available under the MIT License.

