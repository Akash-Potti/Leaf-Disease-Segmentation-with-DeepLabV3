# Leaf Disease Segmentation with DeepLabV3

This code implements leaf disease segmentation using the DeepLabV3 model with a ResNet50 backbone. It loads a dataset from Kaggle containing images of leaves and their corresponding masks indicating disease regions. The dataset is split into training and validation sets, and the model is trained to predict masks for unseen images.

## Dataset
The dataset used in this code can be found on Kaggle: [Leaf Disease Segmentation with Train/Valid Split](https://www.kaggle.com/datasets/sovitrath/leaf-disease-segmentation-with-trainvalid-split). It contains images of leaves along with their corresponding masks indicating diseased regions. The dataset is pre-split into training and validation sets.

## Libraries Used
- `torch`: PyTorch library for deep learning.
- `torchvision`: PyTorch library providing datasets, models, and transformations for computer vision tasks.
- `PIL`: Python Imaging Library for image manipulation.
- `os`: Operating system interface for file operations.
- `numpy`: Library for numerical computing.
- `matplotlib`: Library for creating visualizations.
- `torch.utils.data`: PyTorch library for data loading and manipulation.

## Code Structure
1. **Loading Data**: Loading images and masks from the dataset.
2. **Dataset Class**: Custom dataset class to handle data loading and transformation.
3. **Data Transformation**: Transforming images and masks to the desired format.
4. **Model Definition**: Using DeepLabV3 with ResNet50 backbone for segmentation.
5. **Loss Function and Optimizer**: Defining loss function (BCEWithLogitsLoss) and optimizer (Adam).
6. **Training Loop**: Training the model using the training set.
7. **Validation Loop**: Evaluating the model on the validation set.
8. **Visualization**: Function to visualize sample images, true masks, and generated masks.
9. **Training Process**: Training and validation loop with early stopping mechanism.

## Leaf Disease Segmentation Prediction

This code segment performs leaf disease segmentation prediction using a pre-trained DeepLabV3 model. Given an input image of a leaf, the model predicts the regions of the leaf affected by disease.

### Libraries Used
- `torch`: PyTorch library for deep learning.
- `torchvision`: PyTorch library providing datasets, models, and transformations for computer vision tasks.
- `PIL`: Python Imaging Library for image manipulation.
- `numpy`: Library for numerical computing.
- `matplotlib`: Library for creating visualizations.
- `skimage.measure.find_contours`: Function to find contours in an image.

### Model Loading and Preprocessing
1. **Model Loading**: Load the pre-trained DeepLabV3 model with a ResNet50 backbone.
2. **Adjust Model Output**: Modify the model to output a single channel for binary segmentation.
3. **Load Model Weights**: Load the weights of the best model obtained during training.

### Image Preprocessing
1. **Preprocess Image**: Open the input image, resize it to the desired dimensions, and convert it to a PyTorch tensor.

### Visualization
1. **Visualize Segmentation**: Perform segmentation on the input image and visualize the segmented regions.
2. **Thresholding**: Threshold the segmentation mask to obtain binary segmentation.
3. **Overlay Segmentation**: Overlay the segmented regions on the original image to visualize the affected areas.

### How to Use
1. Ensure that the pre-trained model weights are saved as "best_model.pt".
2. Provide the path to the input image in the `image_path` variable.
3. Run the code to perform leaf disease segmentation prediction.
4. Visualize the segmented regions overlaid on the original image.

### Output
The output consists of two images:
- The original input image.
- The segmented image showing the regions affected by leaf disease highlighted in red.

## Trained Model
The trained model weights can be downloaded from [here](https://drive.google.com/file/d/1Mg1k45JL-jEyNQDcRTe9JipiuDGyAYnP/view?usp=sharing).

This combined markdown documentation provides insights into both the training and prediction processes of leaf disease segmentation using DeepLabV3.
