OmniSeg
A versatile, semi-supervised instance segmentation framework built on PyTorch Lightning. OmniSeg enables training with various state-of-the-art backbones and segmentation heads, including a novel contour-based approach, on the COCO dataset. It's designed for flexibility, allowing researchers and developers to easily experiment with different model architectures and training strategies.

✨ Features
Semi-Supervised Learning (SSL): Leverages a teacher-student model architecture to use both labeled and unlabeled data, a crucial technique for training with limited annotations.

Flexible Backbones: Supports a wide range of popular vision models as backbones, including:

ViT: DINO, SAM

CNN: ResNet, ConvNeXt, RepVGG

Hybrid: Swin Transformer

Modular Segmentation Heads: Experiment with different instance segmentation approaches:

Mask2Former: A state-of-the-art transformer-based segmentation head.

Mask R-CNN: The classic, widely-used two-stage detector.

ContourFormer: A custom head that predicts object contours directly.

PyTorch Lightning: Training and validation are managed by PyTorch Lightning, which simplifies boilerplate code, ensures reproducibility, and supports distributed training.

Automated Data Handling: The project automatically downloads and prepares the COCO 2017 dataset, so you can get started quickly.

🚀 Getting Started
Follow these steps to set up and run OmniSeg.

Prerequisites
Python 3.8+

NVIDIA GPU with CUDA (recommended)

Installation
Clone the repository:

Bash

git clone https://github.com/your-username/omniseg.git
cd omniseg
Install dependencies:

Bash

pip install -r requirements.txt
(Note: You will need to create a requirements.txt file from the imports in your code, or manually install them: pip install transformers torch torchvision tqdm pycocotools pytorch-lightning timm scikit-image scipy)

Prepare the dataset:
The script automatically handles the download and extraction of the COCO 2017 dataset when you first run it. No manual setup is required.

💻 Usage
To train a model, use the train.py script with command-line arguments to configure your desired backbone and segmentation head.

Basic Training
Train a model with a specified backbone and head. The script will automatically download the dataset if it doesn't exist.

Bash

# Example 1: Train a ResNet + Mask R-CNN model
python train.py --backbone resnet --head maskrcnn

# Example 2: Train a DINO ViT + Mask2Former model
python train.py --backbone dino --head mask2former

# Example 3: Train a Swin Transformer + ContourFormer model
python train.py --backbone swin --head contourformer
Advanced Configuration
You can customize training with various parameters.

Argument	Type	Default	Description
--backbone	str	dino	Backbone model (dino, sam, swin, convnext, repvgg, resnet).
--head	str	mask2former	Segmentation head (mask2former, maskrcnn, contourformer).
--learning_rate	float	5e-5	Learning rate.
--image_size	int	None	Custom image size for resizing.
--batch_size	int	None	Batch size per GPU.
--num_labeled_images	int	-1	Number of labeled images to use for training (-1 for all).
--num_unlabeled_images	int	-1	Number of unlabeled images to use for training (-1 for all).
--fast_dev_run	flag	False	Run a single batch for testing the code.
--warmup_steps	int	500	Steps to train on labeled data only before adding unsupervised loss.
--unsup_rampup_steps	int	5000	Steps to ramp up the unsupervised loss weight.

Export to Sheets
Example with advanced settings:

Bash

python train.py --backbone sam --head mask2former --image_size 512 --batch_size 4 --learning_rate 1e-4
🖼️ Visualization
To visualize the predictions of a trained model, use the visualize_model.py script. This script loads a saved checkpoint and displays predictions on random validation images.

Bash

python visualize_model.py path/to/your/checkpoint.ckpt --project_dir ./SSL_Instance_Segmentation
A file named model_predictions_viz.png will be saved in the same directory as your checkpoint.

🤝 Contributing
Contributions are welcome! If you have suggestions for improvements, new backbones, or segmentation heads, please open an issue or submit a pull request.

📝 Attributions
This project makes extensive use of and builds upon the work of others in the open-source community. We would like to acknowledge the following key dependencies and their creators:

PyTorch: An open-source machine learning framework.

PyTorch Lightning: A lightweight PyTorch wrapper for high-performance AI research.

Transformers (Hugging Face): Provides pre-trained models for DINO, SAM, and Swin Transformer.

timm (PyTorch Image Models): A collection of models for computer vision, including ConvNeXt, RepVGG, and ResNet.

COCO Dataset: The Common Objects in Context dataset, providing a benchmark for object detection, segmentation, and captioning.

Mask2Former: A universal segmentation framework.

Mask R-CNN: A widely adopted instance segmentation model.

The semi-supervised learning framework in this project is inspired by the Mean Teacher methodology, which is an effective approach for training with limited labeled data.

Contact: Your Name - https://doloslabs.com
