# Sports ball multi class image classification system using AWS DLAMI (Deep Learning AMI) EC2 Instance
1. Project Overview
This project is a deep learning-based image classification system designed to identify and classify various types of sports balls from images. It was developed and executed on an AWS Deep Learning AMI (DLAMI) using a G5 EC2 instance with GPU acceleration and Jupyter Notebook as the development environment.
Images used for prediction were stored in an Amazon S3 bucket, and the system was capable of displaying the top 3 predicted classes with the highest probabilities for each image.
Key training optimizations implemented in this project include:
•	Transfer Learning
•	Fine-tuning
•	Mixed Precision Training
•	XLA Compilation
•	Efficient Data Pipelines
The deep learning models used include ResNet50 and EfficientNetB0, both trained on a curated dataset of sports ball images.
________________________________________
2. Dataset Description
The dataset contains labeled images of different sports balls. The data is pre-divided into training and validation sets. Key preprocessing steps include:
•	Image resizing
•	Normalization
•	Augmentation (flip, rotate, zoom, color adjustments)
________________________________________
3. Data Preprocessing
Optimized using tf.data API to improve training performance:
•	Batching
•	Caching
•	Prefetching
Augmentation techniques include:
•	Random horizontal/vertical flips
•	Rotations and zoom
•	Brightness/contrast adjustments
________________________________________
4. Model Architecture
-> Transfer Learning
•	Base Models: ResNet50, EfficientNetB0
•	Initial training with frozen layers (feature extraction)
•	Fine-tuning applied by selectively unfreezing top layers
-> Custom Top Layers
•	Global Average Pooling
•	Dense + Dropout Layers
•	Softmax for classification
________________________________________
5. Training Techniques
•	XLA Compilation: Accelerates TensorFlow operations
•	Mixed Precision Training: Reduces memory usage, speeds up training
•	Model Checkpointing: Saves best-performing weights
•	Early Stopping: Prevents overfitting
•	TensorBoard: Real-time monitoring of training metrics
________________________________________
6. Evaluation & Results
•	Models are evaluated using accuracy and loss curves.
•	Visualizations include:
o	Top-3 predicted class probabilities per test image
o	Side-by-side comparison of predictions vs actual labels
EfficientNetB0 showed superior generalization and training speed compared to ResNet50.
