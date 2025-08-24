# YOLOv5s Android Real-Time Detection Training Pipeline
# Complete setup, training, and export for Android deployment

import os
import sys
import subprocess
import shutil
from pathlib import Path
import yaml
import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. INSTALL DEPENDENCIES
# ==========================================

def install_dependencies():
    """Install required packages for YOLOv5 training"""
    print("Installing dependencies...")
    
    # Install basic requirements
    packages = [
        "ultralytics",
        "opencv-python",
        "matplotlib",
        "seaborn",
        "pillow",
        "pyyaml",
        "tqdm",
        "tensorboard",
        "onnx",
        "onnxruntime",
        "tensorflow",
        "tf2onnx"
    ]
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ Installed {package}")
        except subprocess.CalledProcessError:
            print(f"✗ Failed to install {package}")

def clone_yolov5_repo():
    """Clone YOLOv5 repository"""
    print("Cloning YOLOv5 repository...")
    
    if not os.path.exists("yolov5"):
        try:
            subprocess.check_call(["git", "clone", "https://github.com/ultralytics/yolov5.git"])
            print("✓ YOLOv5 repository cloned successfully")
        except subprocess.CalledProcessError:
            print("✗ Failed to clone YOLOv5 repository")
            return False
    else:
        print("✓ YOLOv5 repository already exists")
    
    # Install YOLOv5 requirements
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "yolov5/requirements.txt"])
        print("✓ YOLOv5 requirements installed")
    except subprocess.CalledProcessError:
        print("✗ Failed to install YOLOv5 requirements")
        return False
    
    return True

# ==========================================
# 2. SETUP FOLDER STRUCTURE
# ==========================================

def create_folder_structure():
    """Create the required folder structure for the thesis project"""
    print("Creating folder structure...")
    
    # Define folder structure
    folders = [
        "Thesis/dataset/images/train",
        "Thesis/dataset/images/val", 
        "Thesis/dataset/labels/train",
        "Thesis/dataset/labels/val",
        "Thesis/input",
        "Thesis/output",
        "Thesis/metrics",
        "Thesis/models"
    ]
    
    # Create folders
    for folder in folders:
        Path(folder).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created {folder}")
    
    print("✓ Folder structure created successfully")

def create_dataset_yaml():
    """Create data.yaml file for YOLOv5 training"""
    print("Creating dataset configuration...")
    
    # Dataset configuration
    dataset_config = {
        'train': 'Thesis/dataset/images/train',
        'val': 'Thesis/dataset/images/val',
        'nc': 2,  # Number of classes
        'names': ['nsfw', 'gore']
    }
    
    # Write to yaml file
    with open('Thesis/dataset/data.yaml', 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    print("✓ Dataset configuration file created")
    return 'Thesis/dataset/data.yaml'

# ==========================================
# 3. DATASET PREPARATION UTILITIES
# ==========================================

def validate_dataset_structure():
    """Validate that the dataset structure is correct"""
    print("Validating dataset structure...")
    
    required_paths = [
        'Thesis/dataset/images/train',
        'Thesis/dataset/images/val',
        'Thesis/dataset/labels/train',
        'Thesis/dataset/labels/val',
        'Thesis/dataset/data.yaml'
    ]
    
    for path in required_paths:
        if not os.path.exists(path):
            print(f"✗ Missing: {path}")
            return False
        else:
            print(f"✓ Found: {path}")
    
    # Check for images and labels
    train_images = len(os.listdir('Thesis/dataset/images/train'))
    val_images = len(os.listdir('Thesis/dataset/images/val'))
    train_labels = len(os.listdir('Thesis/dataset/labels/train'))
    val_labels = len(os.listdir('Thesis/dataset/labels/val'))
    
    print(f"Training images: {train_images}, Training labels: {train_labels}")
    print(f"Validation images: {val_images}, Validation labels: {val_labels}")
    
    if train_images == 0 or val_images == 0:
        print("⚠️ Warning: No images found in dataset folders")
        print("Please add your images and labels before training")
        return False
    
    return True

def create_sample_data():
    """Create sample data for demonstration purposes"""
    print("Creating sample dataset for demonstration...")
    
    # Create sample images (blank images with different colors)
    sample_data = [
        ('train', 10),
        ('val', 3)
    ]
    
    for split, count in sample_data:
        for i in range(count):
            # Create sample image
            img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            img_path = f'Thesis/dataset/images/{split}/sample_{i:03d}.jpg'
            cv2.imwrite(img_path, img)
            
            # Create corresponding label
            label_path = f'Thesis/dataset/labels/{split}/sample_{i:03d}.txt'
            with open(label_path, 'w') as f:
                # Sample annotation: class_id x_center y_center width height
                f.write(f"{np.random.randint(0, 2)} {np.random.uniform(0.2, 0.8)} {np.random.uniform(0.2, 0.8)} {np.random.uniform(0.1, 0.3)} {np.random.uniform(0.1, 0.3)}\n")
    
    print("✓ Sample dataset created")

# ==========================================
# 4. TRAINING CONFIGURATION
# ==========================================

class YOLOv5Trainer:
    def __init__(self, data_yaml_path):
        self.data_yaml_path = data_yaml_path
        self.model_size = 'yolov5s'
        self.img_size = 640
        self.batch_size = 16
        self.epochs = 100
        self.project = 'Thesis/training_results'
        self.name = 'yolov5s_training'
        
    def train_model(self):
        """Train YOLOv5s model"""
        print("Starting YOLOv5s training...")
        
        # Change to YOLOv5 directory
        os.chdir('yolov5')
        
        # Training command
        train_cmd = [
            sys.executable, 'train.py',
            '--data', f'../{self.data_yaml_path}',
            '--weights', f'{self.model_size}.pt',
            '--img', str(self.img_size),
            '--batch-size', str(self.batch_size),
            '--epochs', str(self.epochs),
            '--project', f'../{self.project}',
            '--name', self.name,
            '--save-period', '10',  # Save checkpoint every 10 epochs
            '--cache'
        ]
        
        try:
            print(f"Training command: {' '.join(train_cmd)}")
            result = subprocess.run(train_cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✓ Training completed successfully")
                print("Training output:", result.stdout)
            else:
                print("✗ Training failed")
                print("Error:", result.stderr)
                return False
                
        except Exception as e:
            print(f"✗ Training failed with exception: {e}")
            return False
        finally:
            os.chdir('..')  # Return to original directory
            
        return True
    
    def get_best_weights_path(self):
        """Get path to the best trained weights"""
        weights_path = f'{self.project}/{self.name}/weights/best.pt'
        if os.path.exists(weights_path):
            return weights_path
        else:
            print(f"✗ Best weights not found at {weights_path}")
            return None

# ==========================================
# 5. MODEL VALIDATION
# ==========================================

def validate_model(weights_path, data_yaml_path):
    """Validate the trained model"""
    print("Validating trained model...")
    
    os.chdir('yolov5')
    
    val_cmd = [
        sys.executable, 'val.py',
        '--data', f'../{data_yaml_path}',
        '--weights', f'../{weights_path}',
        '--img', '640',
        '--batch-size', '16',
        '--project', '../Thesis/metrics',
        '--name', 'validation_results',
        '--save-txt',
        '--save-conf'
    ]
    
    try:
        print(f"Validation command: {' '.join(val_cmd)}")
        result = subprocess.run(val_cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ Validation completed successfully")
            print("Validation output:", result.stdout)
        else:
            print("✗ Validation failed")
            print("Error:", result.stderr)
            return False
            
    except Exception as e:
        print(f"✗ Validation failed with exception: {e}")
        return False
    finally:
        os.chdir('..')
        
    return True

# ==========================================
# 6. MODEL EXPORT FOR ANDROID
# ==========================================

class ModelExporter:
    def __init__(self, weights_path):
        self.weights_path = weights_path
        self.models_dir = 'Thesis/models'
        
    def export_to_onnx(self):
        """Export YOLOv5 model to ONNX format"""
        print("Exporting model to ONNX...")
        
        os.chdir('yolov5')
        
        onnx_cmd = [
            sys.executable, 'export.py',
            '--weights', f'../{self.weights_path}',
            '--include', 'onnx',
            '--img', '640',
            '--batch-size', '1',
            '--device', 'cpu',
            '--simplify'
        ]
        
        try:
            result = subprocess.run(onnx_cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✓ ONNX export completed successfully")
                
                # Move ONNX file to models directory
                onnx_source = self.weights_path.replace('.pt', '.onnx')
                onnx_dest = f'{self.models_dir}/yolov5s_model.onnx'
                
                if os.path.exists(onnx_source):
                    shutil.move(onnx_source, onnx_dest)
                    print(f"✓ ONNX model saved to {onnx_dest}")
                    return onnx_dest
                else:
                    print("✗ ONNX file not found after export")
                    return None
            else:
                print("✗ ONNX export failed")
                print("Error:", result.stderr)
                return None
                
        except Exception as e:
            print(f"✗ ONNX export failed with exception: {e}")
            return None
        finally:
            os.chdir('..')
    
    def export_to_tflite(self):
        """Export YOLOv5 model to TensorFlow Lite format"""
        print("Exporting model to TensorFlow Lite...")
        
        os.chdir('yolov5')
        
        tflite_cmd = [
            sys.executable, 'export.py',
            '--weights', f'../{self.weights_path}',
            '--include', 'tflite',
            '--img', '640',
            '--batch-size', '1',
            '--device', 'cpu'
        ]
        
        try:
            result = subprocess.run(tflite_cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✓ TensorFlow Lite export completed successfully")
                
                # Move TFLite file to models directory
                tflite_source = self.weights_path.replace('.pt', '.tflite')
                tflite_dest = f'{self.models_dir}/yolov5s_model.tflite'
                
                if os.path.exists(tflite_source):
                    shutil.move(tflite_source, tflite_dest)
                    print(f"✓ TensorFlow Lite model saved to {tflite_dest}")
                    return tflite_dest
                else:
                    print("✗ TensorFlow Lite file not found after export")
                    return None
            else:
                print("✗ TensorFlow Lite export failed")
                print("Error:", result.stderr)
                return None
                
        except Exception as e:
            print(f"✗ TensorFlow Lite export failed with exception: {e}")
            return None
        finally:
            os.chdir('..')

# ==========================================
# 7. MAIN EXECUTION PIPELINE
# ==========================================

def main():
    """Main execution pipeline for YOLOv5s Android training"""
    print("=" * 60)
    print("YOLOv5s Android Real-Time Detection Training Pipeline")
    print("=" * 60)
    
    try:
        # Step 1: Install dependencies
        print("\n### Step 1: Installing Dependencies ###")
        install_dependencies()
        
        if not clone_yolov5_repo():
            print("Failed to clone YOLOv5 repository. Exiting...")
            return
        
        # Step 2: Create folder structure
        print("\n### Step 2: Setting Up Folder Structure ###")
        create_folder_structure()
        data_yaml_path = create_dataset_yaml()
        
        # Step 3: Prepare dataset
        print("\n### Step 3: Preparing Dataset ###")
        if not validate_dataset_structure():
            print("Creating sample dataset for demonstration...")
            create_sample_data()
            
        if not validate_dataset_structure():
            print("Dataset validation failed. Please check your dataset.")
            return
        
        # Step 4: Train model
        print("\n### Step 4: Training YOLOv5s Model ###")
        trainer = YOLOv5Trainer(data_yaml_path)
        
        if not trainer.train_model():
            print("Training failed. Exiting...")
            return
            
        best_weights = trainer.get_best_weights_path()
        if not best_weights:
            print("Could not find trained weights. Exiting...")
            return
        
        print(f"✓ Best weights saved at: {best_weights}")
        
        # Step 5: Validate model
        print("\n### Step 5: Validating Model ###")
        if not validate_model(best_weights, data_yaml_path):
            print("Validation failed, but continuing with export...")
        
        # Step 6: Export for Android
        print("\n### Step 6: Exporting Model for Android ###")
        exporter = ModelExporter(best_weights)
        
        onnx_path = exporter.export_to_onnx()
        tflite_path = exporter.export_to_tflite()
        
        # Final results
        print("\n" + "=" * 60)
        print("TRAINING PIPELINE COMPLETED")
        print("=" * 60)
        
        print("\n📁 Final Exported Files:")
        if onnx_path and os.path.exists(onnx_path):
            print(f"✓ ONNX Model: {os.path.abspath(onnx_path)}")
        else:
            print("✗ ONNX export failed")
            
        if tflite_path and os.path.exists(tflite_path):
            print(f"✓ TensorFlow Lite Model: {os.path.abspath(tflite_path)}")
        else:
            print("✗ TensorFlow Lite export failed")
            
        print(f"\n📊 Training Results: {os.path.abspath(trainer.project)}/{trainer.name}")
        print(f"📈 Validation Metrics: {os.path.abspath('Thesis/metrics/validation_results')}")
        
        print("\n🚀 Models are ready for Android deployment!")
        
        # Display folder structure
        print("\n📂 Complete Project Structure:")
        for root, dirs, files in os.walk("Thesis"):
            level = root.replace("Thesis", "").count(os.sep)
            indent = " " * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = " " * 2 * (level + 1)
            for file in files[:5]:  # Show max 5 files per directory
                print(f"{subindent}{file}")
            if len(files) > 5:
                print(f"{subindent}... and {len(files) - 5} more files")
                
    except Exception as e:
        print(f"\n✗ Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)

# ==========================================
# 8. UTILITY FUNCTIONS
# ==========================================

def check_system_requirements():
    """Check if system meets requirements for training"""
    print("Checking system requirements...")
    
    # Check Python version
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version.major != 3 or python_version.minor != 12:
        print("⚠️ Warning: Python 3.12 is recommended")
    
    # Check PyTorch and CUDA
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
    except ImportError:
        print("PyTorch not installed")
    
    # Check available disk space
    import shutil
    total, used, free = shutil.disk_usage(".")
    print(f"Available disk space: {free // (2**30)} GB")
    
    if free < 5 * (2**30):  # Less than 5GB
        print("⚠️ Warning: Low disk space. Training requires at least 5GB free space")

def create_training_config():
    """Create a training configuration file for easy modification"""
    config = {
        'model': {
            'architecture': 'yolov5s',
            'input_size': 640,
            'num_classes': 2,
            'class_names': ['nsfw', 'gore']
        },
        'training': {
            'batch_size': 16,
            'epochs': 100,
            'learning_rate': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005
        },
        'export': {
            'formats': ['onnx', 'tflite'],
            'optimize_for_mobile': True,
            'quantization': False
        }
    }
    
    with open('Thesis/training_config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print("✓ Training configuration saved to Thesis/training_config.yaml")

if __name__ == "__main__":
    # Run system check first
    check_system_requirements()
    create_training_config()
    
    # Execute main pipeline
    main()
