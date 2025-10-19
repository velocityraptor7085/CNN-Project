"""
Visualization Module for CNN Image Classification

This module provides visualization tools for:
1. Training curves (loss and accuracy over epochs)
2. Confusion matrix heatmap
3. Sample predictions with probabilities
4. Feature map visualizations
5. Learning rate schedule

Mathematical Concepts:
- Training curves show convergence behavior
- Confusion matrix shows classification patterns
- Feature maps reveal what the network learns
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import os
from matplotlib.gridspec import GridSpec


class Visualizer:
    """
    Comprehensive visualization tools for CNN training and evaluation.
    
    Provides:
    - Training history plots
    - Confusion matrix visualization
    - Sample prediction visualization
    - Feature map visualization
    """
    
    def __init__(self, save_dir='./results/plots'):
        """
        Initialize the visualizer.
        
        Args:
            save_dir (str): Directory to save visualizations
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Set style for better-looking plots
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def plot_training_history(self, history, save=True):
        """
        Plot training and validation loss/accuracy over epochs.
        
        Mathematical Interpretation:
        - Loss curve: Shows how error decreases over time
        - Accuracy curve: Shows how performance improves over time
        - Gap between train/val: Indicates overfitting if train >> val
        
        Ideal curves:
        - Loss: Decreasing and converging
        - Accuracy: Increasing and converging
        - Train and Val curves close together (good generalization)
        
        Args:
            history (dict): Training history from Trainer
            save (bool): Whether to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training History', fontsize=16, fontweight='bold')
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # ====================================================================
        # Plot 1: Training and Validation Loss
        # ====================================================================
        axes[0, 0].plot(epochs, history['train_loss'], 'b-o', label='Training Loss', linewidth=2, markersize=4)
        axes[0, 0].plot(epochs, history['val_loss'], 'r-s', label='Validation Loss', linewidth=2, markersize=4)
        axes[0, 0].set_xlabel('Epoch', fontsize=12)
        axes[0, 0].set_ylabel('Loss', fontsize=12)
        axes[0, 0].set_title('Loss over Epochs', fontsize=14)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add annotation for best validation loss
        best_val_idx = np.argmin(history['val_loss'])
        best_val_loss = history['val_loss'][best_val_idx]
        axes[0, 0].annotate(f'Best: {best_val_loss:.4f}',
                          xy=(best_val_idx + 1, best_val_loss),
                          xytext=(10, 10), textcoords='offset points',
                          bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                          arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        # ====================================================================
        # Plot 2: Training and Validation Accuracy
        # ====================================================================
        axes[0, 1].plot(epochs, history['train_acc'], 'b-o', label='Training Accuracy', linewidth=2, markersize=4)
        axes[0, 1].plot(epochs, history['val_acc'], 'r-s', label='Validation Accuracy', linewidth=2, markersize=4)
        axes[0, 1].set_xlabel('Epoch', fontsize=12)
        axes[0, 1].set_ylabel('Accuracy (%)', fontsize=12)
        axes[0, 1].set_title('Accuracy over Epochs', fontsize=14)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add annotation for best validation accuracy
        best_val_idx = np.argmax(history['val_acc'])
        best_val_acc = history['val_acc'][best_val_idx]
        axes[0, 1].annotate(f'Best: {best_val_acc:.2f}%',
                          xy=(best_val_idx + 1, best_val_acc),
                          xytext=(10, -20), textcoords='offset points',
                          bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                          arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        # ====================================================================
        # Plot 3: Learning Rate Schedule
        # ====================================================================
        axes[1, 0].plot(epochs, history['learning_rates'], 'g-o', linewidth=2, markersize=4)
        axes[1, 0].set_xlabel('Epoch', fontsize=12)
        axes[1, 0].set_ylabel('Learning Rate', fontsize=12)
        axes[1, 0].set_title('Learning Rate Schedule', fontsize=14)
        axes[1, 0].set_yscale('log')  # Log scale for better visualization
        axes[1, 0].grid(True, alpha=0.3)
        
        # ====================================================================
        # Plot 4: Overfitting Analysis
        # ====================================================================
        # Gap between training and validation accuracy
        # Large gap indicates overfitting
        gap = np.array(history['train_acc']) - np.array(history['val_acc'])
        axes[1, 1].plot(epochs, gap, 'm-o', linewidth=2, markersize=4)
        axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        axes[1, 1].fill_between(epochs, 0, gap, where=(gap > 0), alpha=0.3, color='red', label='Overfitting')
        axes[1, 1].set_xlabel('Epoch', fontsize=12)
        axes[1, 1].set_ylabel('Train Acc - Val Acc (%)', fontsize=12)
        axes[1, 1].set_title('Overfitting Analysis (Train-Val Gap)', fontsize=14)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            save_path = os.path.join(self.save_dir, 'training_history.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved: {save_path}")
        
        plt.show()
        plt.close()
    
    def plot_confusion_matrix(self, conf_matrix, class_names, save=True):
        """
        Plot confusion matrix as a heatmap.
        
        Mathematical Interpretation:
        - Diagonal elements: Correct predictions
        - Off-diagonal elements: Misclassifications
        - Each row sums to total samples in that class
        
        Reading the matrix:
        - Element [i,j]: Number of class i samples predicted as class j
        - High diagonal values: Good performance
        - High off-diagonal values: Classes being confused
        
        Args:
            conf_matrix (np.ndarray): Confusion matrix
            class_names (list): List of class names
            save (bool): Whether to save the plot
        """
        plt.figure(figsize=(12, 10))
        
        # Normalize confusion matrix by row (actual class)
        # Shows percentage of each class classified as each category
        conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
        
        # Create heatmap
        sns.heatmap(conf_matrix_norm, 
                   annot=True,  # Show numbers in cells
                   fmt='.2%',   # Format as percentage
                   cmap='Blues',  # Color scheme
                   xticklabels=class_names,
                   yticklabels=class_names,
                   cbar_kws={'label': 'Percentage'})
        
        plt.title('Confusion Matrix (Normalized by Actual Class)', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Predicted Class', fontsize=12, fontweight='bold')
        plt.ylabel('Actual Class', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        if save:
            save_path = os.path.join(self.save_dir, 'confusion_matrix.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix plot saved: {save_path}")
        
        plt.show()
        plt.close()
    
    def plot_sample_predictions(self, model, data_loader, class_names, 
                                device='cuda', num_samples=16, save=True):
        """
        Visualize sample predictions with ground truth and probabilities.
        
        Shows:
        - Original image
        - True label
        - Predicted label
        - Prediction confidence
        - Top-3 class probabilities
        
        Args:
            model (nn.Module): Trained model
            data_loader (DataLoader): Data loader
            class_names (list): List of class names
            device (str): Device to use
            num_samples (int): Number of samples to visualize
            save (bool): Whether to save the plot
        """
        model.eval()
        
        # Get a batch of images
        images, labels = next(iter(data_loader))
        images = images[:num_samples]
        labels = labels[:num_samples]
        
        # Make predictions
        with torch.no_grad():
            images_gpu = images.to(device)
            outputs = model(images_gpu)
            probabilities = torch.softmax(outputs, dim=1)
            _, predictions = outputs.max(1)
        
        # Move to CPU for visualization
        probabilities = probabilities.cpu().numpy()
        predictions = predictions.cpu().numpy()
        images = images.cpu()
        labels = labels.cpu().numpy()
        
        # Create subplot grid
        rows = 4
        cols = 4
        fig = plt.figure(figsize=(16, 16))
        gs = GridSpec(rows, cols, figure=fig, hspace=0.4, wspace=0.3)
        
        for idx in range(num_samples):
            ax = fig.add_subplot(gs[idx])
            
            # Denormalize image for display
            # Reverse normalization: x = (x * std) + mean
            mean = np.array([0.4914, 0.4822, 0.4465])
            std = np.array([0.2470, 0.2435, 0.2616])
            img = images[idx].permute(1, 2, 0).numpy()  # CHW -> HWC
            img = img * std + mean
            img = np.clip(img, 0, 1)  # Ensure valid range [0, 1]
            
            # Display image
            ax.imshow(img)
            ax.axis('off')
            
            # Get prediction info
            true_class = class_names[labels[idx]]
            pred_class = class_names[predictions[idx]]
            confidence = probabilities[idx][predictions[idx]]
            
            # Color code: Green if correct, Red if wrong
            color = 'green' if labels[idx] == predictions[idx] else 'red'
            
            # Create title with prediction info
            title = f'True: {true_class}\n'
            title += f'Pred: {pred_class} ({confidence:.2%})'
            ax.set_title(title, fontsize=10, color=color, fontweight='bold')
        
        fig.suptitle('Sample Predictions', fontsize=18, fontweight='bold', y=0.995)
        
        if save:
            save_path = os.path.join(self.save_dir, 'sample_predictions.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Sample predictions plot saved: {save_path}")
        
        plt.show()
        plt.close()
    
    def plot_class_accuracy(self, results, save=True):
        """
        Plot per-class accuracy as a bar chart.
        
        Helps identify:
        - Which classes the model performs well on
        - Which classes need improvement
        - Class imbalance issues
        
        Args:
            results (dict): Evaluation results
            save (bool): Whether to save the plot
        """
        class_names = list(results['per_class'].keys())
        accuracies = [results['per_class'][name]['f1_score'] for name in class_names]
        
        plt.figure(figsize=(12, 6))
        
        # Create bar chart
        bars = plt.bar(range(len(class_names)), accuracies, color='skyblue', edgecolor='navy')
        
        # Color code bars (green if > 70%, orange if 50-70%, red if < 50%)
        for i, bar in enumerate(bars):
            if accuracies[i] >= 70:
                bar.set_color('lightgreen')
            elif accuracies[i] >= 50:
                bar.set_color('orange')
            else:
                bar.set_color('lightcoral')
        
        # Add value labels on bars
        for i, acc in enumerate(accuracies):
            plt.text(i, acc + 1, f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.xlabel('Class', fontsize=12, fontweight='bold')
        plt.ylabel('F1-Score (%)', fontsize=12, fontweight='bold')
        plt.title('Per-Class Performance', fontsize=16, fontweight='bold')
        plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
        plt.ylim(0, 105)
        plt.grid(axis='y', alpha=0.3)
        
        # Add horizontal line for average
        avg_acc = np.mean(accuracies)
        plt.axhline(y=avg_acc, color='r', linestyle='--', label=f'Average: {avg_acc:.2f}%')
        plt.legend()
        
        plt.tight_layout()
        
        if save:
            save_path = os.path.join(self.save_dir, 'class_accuracy.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Class accuracy plot saved: {save_path}")
        
        plt.show()
        plt.close()
    
    def plot_feature_maps(self, model, image, device='cuda', layer_name='conv1', save=True):
        """
        Visualize feature maps from a convolutional layer.
        
        Feature maps show what the network "sees":
        - Early layers: Edges, colors, simple patterns
        - Middle layers: Textures, shapes, object parts
        - Deep layers: High-level object representations
        
        Mathematical Process:
        - Each filter in a conv layer produces one feature map
        - Feature map = result of applying filter to input
        - Bright areas = strong activation (filter detected its pattern)
        
        Args:
            model (nn.Module): Trained model
            image (torch.Tensor): Input image
            device (str): Device to use
            layer_name (str): Layer to visualize ('conv1', 'conv2', 'conv3')
            save (bool): Whether to save the plot
        """
        model.eval()
        
        # Get feature maps
        with torch.no_grad():
            image_gpu = image.unsqueeze(0).to(device)  # Add batch dimension
            feature_maps = model.get_feature_maps(image_gpu, layer_name)
            feature_maps = feature_maps.squeeze(0).cpu().numpy()  # Remove batch dim
        
        # Number of feature maps
        num_maps = feature_maps.shape[0]
        
        # Display up to 64 feature maps
        num_display = min(64, num_maps)
        grid_size = int(np.ceil(np.sqrt(num_display)))
        
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
        fig.suptitle(f'Feature Maps from {layer_name}', fontsize=16, fontweight='bold')
        
        for idx in range(grid_size * grid_size):
            row = idx // grid_size
            col = idx % grid_size
            ax = axes[row, col] if grid_size > 1 else axes
            
            if idx < num_display:
                # Display feature map
                ax.imshow(feature_maps[idx], cmap='viridis')
                ax.set_title(f'Filter {idx}', fontsize=8)
            
            ax.axis('off')
        
        plt.tight_layout()
        
        if save:
            save_path = os.path.join(self.save_dir, f'feature_maps_{layer_name}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature maps plot saved: {save_path}")
        
        plt.show()
        plt.close()


# Example usage
if __name__ == "__main__":
    print("This module should be imported, not run directly.")
    print("See main.py for complete visualization pipeline.")
