# YOLOv5 Model Conversion for Android Detection of NSFW in Real Time

A sample Python implementation for training YOLOv5 models and exporting them for Android deployment. This implementation provides an end-to-end pipeline from dataset preparation to model export, with built-in support for training on custom datasets or using preexisting datasets like COCO128.

## Features

- **Automated Environment Setup**: System requirement checks and dependency installation
- **Flexible Dataset Support**: Works with custom datasets or falls back to COCO128 for demonstration
- **Complete Training Pipeline**: YOLOv5s model training with configurable parameters
- **Model Validation**: Built-in validation and metrics generation
- **Multi-format Export**: Export trained models to ONNX, TensorFlow Lite, and Keras formats
- **Visualization Tools**: Training metrics visualization and result analysis
- **Android-Ready Models**: Optimized exports specifically for mobile deployment

## System Requirements

- Python 3.8 or higher
- PyTorch with CUDA support (recommended)
- At least 5GB free disk space
- Git (for cloning YOLOv5 repository)

## Installation

The script automatically handles dependency installation, but you can manually install requirements:

```bash
pip install ultralytics opencv-python matplotlib seaborn pillow pyyaml tqdm tensorboard onnx onnxruntime tensorflow tf2onnx
```

## Project Structure

```
Thesis/
├── dataset/
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   ├── labels/
│   │   ├── train/
│   │   └── val/
│   └── data.yaml
├── input/
├── output/
├── metrics/
├── models/
├── training_results/
└── training_config.yaml
```

## Usage

### Basic Usage

Simply run the script to start the complete training pipeline:

```python
python yolov5s.ipynb
```

The script will automatically:
1. Check system requirements
2. Install dependencies
3. Clone YOLOv5 repository
4. Set up folder structure
5. Validate or prepare dataset
6. Train the model
7. Validate the trained model
8. Export to multiple formats

### Custom Dataset

To use your own dataset:

1. Place training images in `Thesis/dataset/images/train/`
2. Place validation images in `Thesis/dataset/images/val/`
3. Place corresponding YOLO format labels in `Thesis/dataset/labels/train/` and `Thesis/dataset/labels/val/`
4. Update the class names and count in `data.yaml`

### Configuration

Modify training parameters in `Thesis/training_config.yaml`:

```yaml
model:
  architecture: yolov5s
  input_size: 640
  num_classes: 2
  class_names: ['nsfw', 'gore']

training:
  batch_size: 16
  epochs: 100
  learning_rate: 0.01
  momentum: 0.937
  weight_decay: 0.0005
  patience: 15

export:
  formats: ['onnx', 'tflite']
  optimize_for_mobile: true
  quantization: true
```

## Key Classes and Functions

### YOLOv5Trainer
Main training class that handles model training with configurable parameters.

```python
trainer = YOLOv5Trainer(data_yaml_path, use_preexisting=False)
training_success = trainer.train_model()
```

### ModelExporter
Handles exporting trained models to different formats for deployment.

```python
exporter = ModelExporter(weights_path)
onnx_path = exporter.export_to_onnx()
tflite_path = exporter.export_to_tflite()
```

## Dataset Support

### Custom Dataset Format
- Images: JPG, PNG, BMP formats
- Labels: YOLO format (.txt files)
- Structure: Separate train/val splits

### Automatic Fallback
If no custom dataset is detected, the script automatically downloads and configures COCO128 for demonstration purposes.

## Model Exports

The script exports trained models in multiple formats:

- **ONNX**: For cross-platform deployment
- **TensorFlow Lite**: Optimized for mobile devices with INT8 quantization
- **Keras**: Native TensorFlow format (if supported)

## Visualization and Analysis

The script includes comprehensive visualization tools:

- Training loss curves
- mAP (Mean Average Precision) metrics
- Precision and Recall curves
- Confusion matrices
- Sample predictions
- Learning rate schedules

## Output Files

After successful training, you'll find:

- **Trained weights**: `Thesis/training_results/yolov5s_training/weights/best.pt`
- **Exported models**: `Thesis/models/`
- **Training metrics**: `Thesis/training_results/yolov5s_training/`
- **Validation results**: `Thesis/metrics/validation_results/`
- **Inference results**: `Thesis/output/inference_results/`

## Android Deployment

The exported models are optimized for Android deployment:

1. **ONNX Model**: Use with ONNX Runtime for Android
2. **TensorFlow Lite Model**: Integrate with TensorFlow Lite Android library
3. **Quantized INT8**: Reduced model size for mobile devices

## Troubleshooting

### Common Issues

1. **CUDA not available**: The script automatically falls back to CPU training
2. **Low disk space**: Ensure at least 5GB free space
3. **Dataset not found**: Script automatically uses COCO128 as fallback
4. **Export failures**: Check PyTorch and TensorFlow versions

### Debug Functions

The script includes debug utilities:
- `debug_dataset_paths()`: Validates dataset structure
- `check_system_requirements()`: Verifies system setup
- `validate_dataset_structure()`: Checks dataset integrity

## Performance Notes

- **GPU Training**: Significantly faster than CPU
- **Batch Size**: Adjust based on available GPU memory
- **Epochs**: Start with 50-100 epochs, adjust based on convergence
- **Model Size**: YOLOv5s provides good balance of speed and accuracy

## Dependencies

Core dependencies automatically installed:
- ultralytics
- opencv-python
- matplotlib
- seaborn
- pillow
- pyyaml
- tqdm
- tensorboard
- onnx
- onnxruntime
- tensorflow
- tf2onnx

## License

This project builds upon the YOLOv5 repository by Ultralytics. Please refer to their licensing terms for commercial usage.

## Contributing

Contributions are welcome! Please ensure:
- Code follows existing style conventions
- New features include appropriate error handling
- Documentation is updated for new functionality

## Support

For issues and questions:
1. Check the troubleshooting section
2. Verify system requirements
3. Ensure proper dataset structure
4. Review YOLOv5 documentation for training-specific issues
