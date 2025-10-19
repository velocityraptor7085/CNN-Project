"""
Main Execution Script for CNN Image Classification

This script orchestrates the complete deep learning pipeline:
1. Data Loading and Preprocessing
2. Model Creation and Initialization
3. Model Training
4. Model Evaluation
5. Results Visualization

Educational Purpose:
This script demonstrates a complete end-to-end deep learning workflow,
from raw data to trained model with comprehensive analysis.

Usage:
    python main.py [--epochs EPOCHS] [--batch-size BATCH_SIZE] [--lr LEARNING_RATE]

Mathematical Pipeline:
Data → Preprocessing → Model → Training (Forward + Backward + Update) → Evaluation → Insights
"""

import torch
import argparse
import os
import sys

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_preprocessing import DataPreprocessor
from src.model import CNN
from src.train import Trainer
from src.evaluate import Evaluator
from src.visualization import Visualizer


def parse_arguments():
    """
    Parse command-line arguments for configurable training.
    
    Hyperparameters that can be tuned:
    - epochs: Number of training iterations through dataset
    - batch_size: Number of samples per gradient update
    - learning_rate: Step size for weight updates
    - dropout: Probability of dropping neurons (regularization)
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='CNN Image Classification for CIFAR-10',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=20,
                      help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                      help='Batch size for training and evaluation')
    parser.add_argument('--lr', type=float, default=0.001,
                      help='Learning rate for optimizer')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                      help='Weight decay (L2 regularization)')
    parser.add_argument('--dropout', type=float, default=0.5,
                      help='Dropout rate for regularization')
    
    # Data parameters
    parser.add_argument('--val-split', type=float, default=0.1,
                      help='Validation split ratio')
    parser.add_argument('--num-workers', type=int, default=2,
                      help='Number of data loading workers')
    
    # Model parameters
    parser.add_argument('--resume', type=str, default=None,
                      help='Path to checkpoint to resume training')
    
    # Other settings
    parser.add_argument('--no-cuda', action='store_true',
                      help='Disable CUDA even if available')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    
    return parser.parse_args()


def set_seed(seed):
    """
    Set random seeds for reproducibility.
    
    Why reproducibility matters:
    - Scientific experiments should be repeatable
    - Debugging is easier with deterministic behavior
    - Fair comparison of different approaches
    
    Mathematical Note:
    - Random number generators use seeds to produce sequences
    - Same seed = same sequence of "random" numbers
    - Different seeds = different sequences
    
    Args:
        seed (int): Random seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np
    np.random.seed(seed)
    import random
    random.seed(seed)
    
    # Make PyTorch operations deterministic
    # Note: This may reduce performance
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_system_info(device):
    """
    Print system and PyTorch configuration information.
    
    Helps understand:
    - Available hardware (GPU/CPU)
    - PyTorch version and capabilities
    - Memory constraints
    
    Args:
        device (str): Device being used ('cuda' or 'cpu')
    """
    print("\n" + "="*70)
    print("SYSTEM INFORMATION")
    print("="*70)
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Device: {device}")
    
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        # GPU memory information
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"Total GPU Memory: {total_memory:.2f} GB")
    else:
        print("Running on CPU (training will be slower)")
        print("Tip: Install CUDA-enabled PyTorch for GPU acceleration")
    
    print("="*70)


def main():
    """
    Main execution function that runs the complete pipeline.
    
    Pipeline Steps:
    1. Setup: Parse arguments, set seed, check device
    2. Data: Load and preprocess CIFAR-10 dataset
    3. Model: Create CNN architecture
    4. Train: Train model using backpropagation
    5. Evaluate: Assess model performance on test set
    6. Visualize: Create plots and analysis
    
    Mathematical Flow:
    Raw Images → Normalized Data → CNN(θ) → Loss → ∇L → θ_new → Predictions → Metrics
    """
    # ========================================================================
    # STEP 1: SETUP AND CONFIGURATION
    # ========================================================================
    
    print("\n" + "="*70)
    print("CNN IMAGE CLASSIFICATION PROJECT")
    print("Educational Deep Learning Implementation")
    print("="*70)
    
    # Parse command-line arguments
    args = parse_arguments()
    
    # Set random seed for reproducibility
    # This ensures experiments can be replicated exactly
    set_seed(args.seed)
    print(f"\n✓ Random seed set to: {args.seed}")
    
    # Determine device (GPU vs CPU)
    # GPU is much faster for deep learning (10-100x speedup)
    # Mathematical operations like matrix multiplication are parallelized on GPU
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'
    
    # Print system information
    print_system_info(device)
    
    # ========================================================================
    # STEP 2: DATA LOADING AND PREPROCESSING
    # ========================================================================
    
    print("\n" + "="*70)
    print("STEP 1: DATA LOADING AND PREPROCESSING")
    print("="*70)
    
    # Initialize data preprocessor
    # Handles normalization, augmentation, and batching
    preprocessor = DataPreprocessor(
        data_dir='./data',
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Load CIFAR-10 dataset
    # Mathematical transformations:
    # 1. Normalize: (x - μ) / σ  [Makes training more stable]
    # 2. Augment: Random flips, crops, rotations  [Prevents overfitting]
    # 3. Batch: Group samples  [Efficient GPU processing]
    train_loader, val_loader, test_loader = preprocessor.load_data(
        val_split=args.val_split
    )
    
    print(f"\n✓ Data loaded successfully")
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # ========================================================================
    # STEP 3: MODEL CREATION
    # ========================================================================
    
    print("\n" + "="*70)
    print("STEP 2: MODEL CREATION")
    print("="*70)
    
    # Create CNN model
    # Architecture: Conv layers (feature extraction) + FC layers (classification)
    # Mathematical operations:
    # - Convolution: Local feature detection
    # - ReLU: Non-linear activation
    # - Pooling: Dimensionality reduction
    # - Batch Norm: Training stabilization
    # - Dropout: Regularization
    model = CNN(num_classes=10, dropout_rate=args.dropout)
    
    # Print model summary
    total_params = model.count_parameters()
    print(f"\n✓ Model created successfully")
    print(f"  Architecture: CNN with 3 convolutional blocks")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Model size: {total_params * 4 / 1e6:.2f} MB")
    
    # Print model architecture
    print("\nModel Architecture:")
    print("-" * 70)
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        print(f"  {name:20s}: {params:>10,} parameters")
    print("-" * 70)
    
    # ========================================================================
    # STEP 4: MODEL TRAINING
    # ========================================================================
    
    print("\n" + "="*70)
    print("STEP 3: MODEL TRAINING")
    print("="*70)
    
    # Initialize trainer
    # Mathematical process:
    # 1. Forward: ŷ = f(x; θ)
    # 2. Loss: L = CrossEntropy(ŷ, y)
    # 3. Backward: ∇θL via chain rule (backpropagation)
    # 4. Update: θ ← θ - α∇θL (gradient descent)
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Load checkpoint if resuming training
    if args.resume:
        print(f"\nResuming training from: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Train the model
    # This is where the magic happens!
    # The model learns to recognize patterns through:
    # - Forward propagation: Computing predictions
    # - Loss calculation: Measuring errors
    # - Backpropagation: Computing gradients
    # - Weight updates: Improving parameters
    print(f"\nStarting training for {args.epochs} epochs...")
    print("This may take several minutes depending on your hardware.\n")
    
    history = trainer.train(
        num_epochs=args.epochs,
        checkpoint_dir='./models/checkpoints'
    )
    
    # ========================================================================
    # STEP 5: MODEL EVALUATION
    # ========================================================================
    
    print("\n" + "="*70)
    print("STEP 4: MODEL EVALUATION")
    print("="*70)
    
    # Load best model for evaluation
    # We use the model with highest validation accuracy
    # This is the model that generalizes best to unseen data
    if trainer.best_model_path:
        print(f"\nLoading best model from: {trainer.best_model_path}")
        checkpoint = torch.load(trainer.best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Initialize evaluator
    # Computes comprehensive metrics:
    # - Accuracy: Overall correctness
    # - Precision: TP / (TP + FP)
    # - Recall: TP / (TP + FN)
    # - F1-Score: Harmonic mean of precision and recall
    evaluator = Evaluator(
        model=model,
        test_loader=test_loader,
        device=device,
        class_names=preprocessor.classes
    )
    
    # Evaluate on test set
    print("\nEvaluating model on test set...")
    results = evaluator.evaluate()
    
    # Print results
    evaluator.print_results(results)
    
    # Analyze errors
    print("\n" + "="*70)
    print("ERROR ANALYSIS")
    print("="*70)
    error_analysis = evaluator.analyze_errors(results)
    
    # Save results
    evaluator.save_results(results)
    
    # ========================================================================
    # STEP 6: VISUALIZATION
    # ========================================================================
    
    print("\n" + "="*70)
    print("STEP 5: VISUALIZATION")
    print("="*70)
    
    # Initialize visualizer
    visualizer = Visualizer(save_dir='./results/plots')
    
    print("\nGenerating visualizations...")
    
    # Plot training history
    # Shows how loss and accuracy evolved during training
    # Helps diagnose training issues (overfitting, underfitting, etc.)
    print("  → Plotting training history...")
    visualizer.plot_training_history(history)
    
    # Plot confusion matrix
    # Shows which classes are confused with each other
    # Diagonal = correct predictions, Off-diagonal = mistakes
    print("  → Plotting confusion matrix...")
    import numpy as np
    conf_matrix = np.array(results['confusion_matrix'])
    visualizer.plot_confusion_matrix(conf_matrix, preprocessor.classes)
    
    # Plot class-wise accuracy
    # Shows performance for each individual class
    # Helps identify which classes are harder to classify
    print("  → Plotting per-class accuracy...")
    visualizer.plot_class_accuracy(results)
    
    # Plot sample predictions
    # Visual check of model predictions
    # Shows actual images with predictions and confidence
    print("  → Plotting sample predictions...")
    visualizer.plot_sample_predictions(
        model=model,
        data_loader=test_loader,
        class_names=preprocessor.classes,
        device=device,
        num_samples=16
    )
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    
    print("\n" + "="*70)
    print("PROJECT COMPLETE!")
    print("="*70)
    
    print(f"\nFinal Results:")
    print(f"  ✓ Test Accuracy: {results['overall']['accuracy']:.2f}%")
    print(f"  ✓ Best Validation Accuracy: {trainer.best_val_acc:.2f}%")
    print(f"  ✓ Total Training Epochs: {len(history['train_loss'])}")
    
    print(f"\nSaved Artifacts:")
    print(f"  ✓ Best model: {trainer.best_model_path}")
    print(f"  ✓ Training history: ./models/checkpoints/training_history.json")
    print(f"  ✓ Evaluation results: ./results/metrics/evaluation_results.json")
    print(f"  ✓ Visualizations: ./results/plots/")
    
    print("\n" + "="*70)
    print("Thank you for using this educational CNN implementation!")
    print("Check PROJECT_REPORT.md for detailed mathematical explanations.")
    print("="*70 + "\n")


if __name__ == "__main__":
    """
    Entry point for the script.
    
    When you run: python main.py
    This code executes and starts the complete pipeline.
    
    The pipeline demonstrates:
    1. Data preprocessing (normalization, augmentation)
    2. Model architecture (CNNs for computer vision)
    3. Training (backpropagation, gradient descent)
    4. Evaluation (metrics, confusion matrix)
    5. Visualization (understanding results)
    
    Mathematical Journey:
    Raw Pixels → Normalized Features → Learned Representations → Class Predictions
    
    This is the essence of deep learning for image classification!
    """
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        print("Progress has been saved. You can resume training with --resume flag.")
    except Exception as e:
        print(f"\n\nError occurred: {str(e)}")
        print("Check the error message above for details.")
        import traceback
        traceback.print_exc()
