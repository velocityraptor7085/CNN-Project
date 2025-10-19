# Quick Start Guide

This guide will help you get the CNN project up and running quickly.

## Prerequisites

- Python 3.8 or higher
- (Optional but recommended) NVIDIA GPU with CUDA support

## Installation Steps

### 1. Create and Activate Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

**Linux/Mac:**
```bash
python -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```powershell
pip install -r requirements.txt
```

This will install:
- PyTorch (deep learning framework)
- torchvision (computer vision utilities)
- NumPy (numerical computing)
- Matplotlib/Seaborn (visualization)
- scikit-learn (metrics)
- And other required packages

**Note**: If you have a CUDA-enabled GPU, visit [pytorch.org](https://pytorch.org) to install the GPU version of PyTorch for faster training.

## Running the Project

### Basic Usage

Run with default settings (20 epochs, batch size 64):
```powershell
python main.py
```

### Custom Settings

Train for 30 epochs with batch size 128:
```powershell
python main.py --epochs 30 --batch-size 128
```

Adjust learning rate:
```powershell
python main.py --lr 0.0001
```

Use CPU only (no GPU):
```powershell
python main.py --no-cuda
```

### All Available Options

```
--epochs EPOCHS           Number of training epochs (default: 20)
--batch-size BATCH_SIZE   Batch size (default: 64)
--lr LEARNING_RATE        Learning rate (default: 0.001)
--weight-decay DECAY      L2 regularization (default: 1e-4)
--dropout RATE            Dropout probability (default: 0.5)
--val-split SPLIT         Validation split ratio (default: 0.1)
--num-workers WORKERS     Data loading workers (default: 2)
--resume PATH             Resume from checkpoint
--no-cuda                 Disable CUDA
--seed SEED               Random seed (default: 42)
```

## Expected Output

When you run the project, you'll see:

1. **System Information**: GPU/CPU details
2. **Data Loading**: CIFAR-10 dataset download and preprocessing
3. **Model Summary**: Architecture and parameter count
4. **Training Progress**: Loss and accuracy for each epoch
5. **Evaluation Results**: Test set performance metrics
6. **Visualizations**: Training curves, confusion matrix, sample predictions

## Output Files

After running, check these directories:

- `models/checkpoints/`: Saved model weights
- `results/plots/`: Visualization images
- `results/metrics/`: Evaluation metrics (JSON)
- `data/`: Downloaded CIFAR-10 dataset

## Training Time

Approximate training times for 20 epochs:

- **CPU**: 2-3 hours
- **GPU (GTX 1060/1660)**: 15-20 minutes
- **GPU (RTX 3070/3080)**: 5-10 minutes

## Testing Individual Modules

Test data preprocessing:
```powershell
python src/data_preprocessing.py
```

Test model architecture:
```powershell
python src/model.py
```

## Troubleshooting

### Issue: "Import torch could not be resolved"
**Solution**: Install PyTorch: `pip install torch torchvision`

### Issue: CUDA out of memory
**Solution**: Reduce batch size: `python main.py --batch-size 32`

### Issue: Slow training on CPU
**Solution**: Either use GPU or reduce epochs/batch size

### Issue: Low accuracy
**Solution**: 
- Train for more epochs
- Adjust learning rate
- Check if data augmentation is working

## Next Steps

1. ✅ Run the default training
2. 📊 Check the visualizations in `results/plots/`
3. 📖 Read `PROJECT_REPORT.md` for detailed explanations
4. 🔧 Experiment with different hyperparameters
5. 🚀 Try modifying the model architecture

## Learning Resources

- **Code Comments**: Every file has extensive educational comments
- **PROJECT_REPORT.md**: Comprehensive mathematical explanations
- **README.md**: Project overview and structure

## Getting Help

Common questions:
- **What is CIFAR-10?** A dataset of 60,000 32x32 color images in 10 classes
- **What is a CNN?** A neural network specialized for image processing
- **Why GPU?** 10-100x faster than CPU for deep learning
- **What's a good accuracy?** 75-85% is expected for this architecture

Happy Learning! 🎓🚀
