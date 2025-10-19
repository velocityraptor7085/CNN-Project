"""
Training Module for CNN Image Classification

This module handles the complete training process including:
1. Forward propagation (computing predictions)
2. Loss calculation (measuring prediction error)
3. Backpropagation (computing gradients)
4. Weight updates (optimizing model parameters)
5. Validation (monitoring generalization)

Mathematical Concepts:
- Loss Function: Measures difference between predictions and ground truth
- Gradient Descent: Iterative optimization algorithm
- Backpropagation: Efficient gradient computation using chain rule
- Learning Rate: Step size for weight updates
- Momentum: Accelerates gradient descent in relevant directions
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time
import os
import json


class Trainer:
    """
    Handles the training and validation of the CNN model.
    
    Training Process:
    1. Forward Pass: Compute predictions
    2. Calculate Loss: Measure error
    3. Backward Pass: Compute gradients
    4. Update Weights: Apply optimizer
    5. Validate: Check performance on validation set
    6. Save Checkpoint: Store best model
    """
    
    def __init__(self, model, train_loader, val_loader, device='cuda', 
                 learning_rate=0.001, weight_decay=1e-4):
        """
        Initialize the trainer.
        
        Args:
            model (nn.Module): CNN model to train
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader
            device (str): Device to use ('cuda' or 'cpu')
            learning_rate (float): Initial learning rate for optimizer
            weight_decay (float): L2 regularization strength
        
        Mathematical Notes:
        - Learning Rate: Controls step size in gradient descent
            - Too high: Training unstable, might diverge
            - Too low: Training very slow, might get stuck
            - Typical range: 0.0001 to 0.1
        
        - Weight Decay (L2 Regularization): Prevents overfitting
            - Formula: Loss = CrossEntropy + λ * Σ(w²)
            - Penalizes large weights
            - Typical range: 1e-5 to 1e-3
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.learning_rate = learning_rate
        
        # Move model to GPU if available
        # GPU benefits:
        # - Parallel processing of matrix operations
        # - Much faster training (10-100x speedup)
        # - Handles larger batches
        self.model.to(self.device)
        
        # ========================================================================
        # LOSS FUNCTION: Cross-Entropy Loss
        # ========================================================================
        
        # Cross-Entropy Loss for multi-class classification
        # Mathematical Formula:
        # L = -Σ(y_true * log(y_pred))
        # 
        # For a single sample with true class c:
        # L = -log(P(class = c)) = -log(exp(z_c) / Σ(exp(z_i)))
        # 
        # Where:
        # - z_i: Raw output scores (logits) from the network
        # - P(class = c): Probability of correct class after softmax
        #
        # Why Cross-Entropy?
        # 1. Suitable for probabilistic interpretation
        # 2. Penalizes confident wrong predictions heavily
        # 3. Gradient descent converges well with this loss
        # 4. Standard choice for classification tasks
        #
        # Note: PyTorch's CrossEntropyLoss combines softmax and negative log-likelihood
        # So we don't apply softmax in the model output!
        self.criterion = nn.CrossEntropyLoss()
        
        # ========================================================================
        # OPTIMIZER: Adam (Adaptive Moment Estimation)
        # ========================================================================
        
        # Adam Optimizer - combines benefits of momentum and adaptive learning rates
        # Mathematical Formula:
        # m_t = β₁ * m_{t-1} + (1 - β₁) * g_t          (First moment - mean)
        # v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²         (Second moment - variance)
        # m̂_t = m_t / (1 - β₁ᵗ)                        (Bias correction)
        # v̂_t = v_t / (1 - β₂ᵗ)                        (Bias correction)
        # θ_t = θ_{t-1} - α * m̂_t / (√v̂_t + ε)        (Weight update)
        #
        # Where:
        # - g_t: Gradient at time t
        # - m_t: First moment estimate (running average of gradients)
        # - v_t: Second moment estimate (running average of squared gradients)
        # - α: Learning rate (how fast we update)
        # - β₁, β₂: Decay rates for moments (default: 0.9, 0.999)
        # - ε: Small constant for numerical stability (default: 1e-8)
        #
        # Why Adam?
        # 1. Adapts learning rate for each parameter (some learn faster than others)
        # 2. Handles sparse gradients well
        # 3. Relatively insensitive to hyperparameter choices
        # 4. Combines momentum (faster convergence) and RMSprop (adaptive rates)
        # 5. Works well in practice for most deep learning tasks
        #
        # Alternative optimizers:
        # - SGD: Simple but needs careful tuning
        # - SGD + Momentum: Better than SGD, still needs tuning
        # - RMSprop: Good for RNNs
        # - AdaGrad: Good for sparse data
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),      # Decay rates for first and second moments
            eps=1e-8,                # For numerical stability
            weight_decay=weight_decay # L2 regularization
        )
        
        # ========================================================================
        # LEARNING RATE SCHEDULER
        # ========================================================================
        
        # ReduceLROnPlateau: Reduces learning rate when validation loss plateaus
        # Mathematical Strategy:
        # - If val_loss doesn't improve for 'patience' epochs
        # - Multiply learning rate by 'factor'
        # - New_LR = Old_LR * factor
        #
        # Why reduce learning rate?
        # - High LR early: Fast initial progress
        # - Lower LR later: Fine-tune to find better minimum
        # - Like zooming in on the optimal solution
        #
        # Parameters:
        # - mode='min': We want to minimize validation loss
        # - factor=0.5: Reduce LR by half when plateau detected
        # - patience=3: Wait 3 epochs before reducing
        # - verbose=True: Print when LR is reduced
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True,
            min_lr=1e-6  # Don't go below this learning rate
        )
        
        # Training history tracking
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }
        
        # Best model tracking
        self.best_val_acc = 0.0
        self.best_model_path = None
    
    def train_epoch(self):
        """
        Train for one epoch (one complete pass through training data).
        
        Training Process for Each Batch:
        1. Forward Pass: Compute predictions
        2. Calculate Loss: Compare predictions with ground truth
        3. Backward Pass: Compute gradients using backpropagation
        4. Update Weights: Apply optimizer to update parameters
        
        Mathematical Process:
        1. Forward: ŷ = f(x; θ)  [Prediction using current weights]
        2. Loss: L = CrossEntropy(ŷ, y)  [Measure error]
        3. Backward: ∇θL = ∂L/∂θ  [Compute gradients via chain rule]
        4. Update: θ_new = θ_old - α * ∇θL  [Gradient descent step]
        
        Returns:
            tuple: (average_loss, accuracy)
        """
        # Set model to training mode
        # Why?
        # - Enables dropout (randomly drops neurons)
        # - Enables batch normalization training mode (uses batch statistics)
        self.model.train()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Progress bar for visual feedback
        pbar = tqdm(self.train_loader, desc='Training', leave=False)
        
        for batch_idx, (images, labels) in enumerate(pbar):
            # Move data to GPU
            # Images: [batch_size, 3, 32, 32]
            # Labels: [batch_size]
            images, labels = images.to(self.device), labels.to(self.device)
            
            # ====================================================================
            # STEP 1: ZERO GRADIENTS
            # ====================================================================
            # Clear previous gradients
            # Why? PyTorch accumulates gradients by default
            # We want fresh gradients for each batch
            self.optimizer.zero_grad()
            
            # ====================================================================
            # STEP 2: FORWARD PASS
            # ====================================================================
            # Compute predictions
            # outputs: [batch_size, num_classes] - raw scores (logits)
            # Mathematical Flow:
            # Input → Conv layers → Pooling → FC layers → Output (logits)
            outputs = self.model(images)
            
            # ====================================================================
            # STEP 3: COMPUTE LOSS
            # ====================================================================
            # Calculate cross-entropy loss
            # Mathematical Formula:
            # L = -Σ(y_true * log(softmax(y_pred)))
            # 
            # The loss measures how far our predictions are from the truth
            # - Lower loss = better predictions
            # - Higher loss = worse predictions
            loss = self.criterion(outputs, labels)
            
            # ====================================================================
            # STEP 4: BACKWARD PASS (Backpropagation)
            # ====================================================================
            # Compute gradients using chain rule
            # Mathematical Process:
            # 1. Start with ∂L/∂output (loss gradient w.r.t. output)
            # 2. Propagate backward through each layer:
            #    ∂L/∂w = ∂L/∂output * ∂output/∂w (chain rule)
            # 3. Store gradients in param.grad for each parameter
            #
            # Chain Rule Example for a simple network:
            # If y = f(g(x)), then dy/dx = dy/dg * dg/dx
            #
            # For deep networks:
            # ∂L/∂w₁ = ∂L/∂y * ∂y/∂h₃ * ∂h₃/∂h₂ * ∂h₂/∂h₁ * ∂h₁/∂w₁
            # 
            # PyTorch automatically computes all these derivatives!
            loss.backward()
            
            # ====================================================================
            # STEP 5: GRADIENT CLIPPING (Optional but helpful)
            # ====================================================================
            # Prevents exploding gradients
            # If gradients are too large, training becomes unstable
            # Clip gradients to maximum norm of 1.0
            # Mathematical: If ||g|| > max_norm, then g = g * (max_norm / ||g||)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # ====================================================================
            # STEP 6: UPDATE WEIGHTS
            # ====================================================================
            # Apply optimizer to update model parameters
            # Mathematical (for Adam):
            # θ_new = θ_old - α * m̂ / (√v̂ + ε)
            # where m̂ and v̂ are bias-corrected moment estimates
            #
            # This is where the model actually learns!
            self.optimizer.step()
            
            # ====================================================================
            # TRACKING METRICS
            # ====================================================================
            
            # Accumulate loss for averaging
            running_loss += loss.item()
            
            # Calculate accuracy
            # Get predicted class (index of maximum logit)
            # Mathematical: predicted = argmax(outputs)
            _, predicted = outputs.max(1)
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'acc': 100. * correct / total
            })
        
        # Calculate epoch statistics
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        """
        Validate the model on validation set.
        
        Validation vs Training:
        - No gradient computation (saves memory and computation)
        - No weight updates (just evaluation)
        - Dropout disabled (use all neurons)
        - Batch norm uses running statistics (not batch statistics)
        
        Why validate?
        - Monitor generalization (how well model works on unseen data)
        - Detect overfitting (training acc ↑, validation acc ↓)
        - Guide hyperparameter tuning
        - Determine when to stop training (early stopping)
        
        Returns:
            tuple: (average_loss, accuracy)
        """
        # Set model to evaluation mode
        # Effects:
        # - Disables dropout (uses all neurons with scaled outputs)
        # - Batch norm uses running mean/std instead of batch statistics
        self.model.eval()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Disable gradient computation
        # Why?
        # - Saves memory (don't need to store intermediate values)
        # - Faster computation (no backward pass)
        # - We're only evaluating, not training
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation', leave=False)
            
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Forward pass only (no backward pass in validation)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Track metrics
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': loss.item(),
                    'acc': 100. * correct / total
                })
        
        # Calculate validation statistics
        val_loss = running_loss / len(self.val_loader)
        val_acc = 100. * correct / total
        
        return val_loss, val_acc
    
    def train(self, num_epochs, checkpoint_dir='./models/checkpoints'):
        """
        Train the model for specified number of epochs.
        
        Training Loop:
        For each epoch:
            1. Train on training set
            2. Validate on validation set
            3. Update learning rate based on validation performance
            4. Save best model checkpoint
            5. Track metrics
        
        Args:
            num_epochs (int): Number of complete passes through training data
            checkpoint_dir (str): Directory to save model checkpoints
        
        Mathematical Concept - Epoch:
        - 1 epoch = 1 complete pass through entire training dataset
        - More epochs = more training, but risk of overfitting
        - Typical range: 10-200 epochs depending on dataset size
        
        Returns:
            dict: Training history
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        print("="*60)
        print(f"Starting Training for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Initial Learning Rate: {self.learning_rate}")
        print(f"Optimizer: {self.optimizer.__class__.__name__}")
        print("="*60)
        
        start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            epoch_start_time = time.time()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            print(f"\nEpoch [{epoch}/{num_epochs}] - LR: {current_lr:.6f}")
            
            # Train for one epoch
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Update learning rate based on validation loss
            # If validation loss plateaus, reduce learning rate
            # This helps fine-tune the model to find better minima
            self.scheduler.step(val_loss)
            
            # Track metrics
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(current_lr)
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time
            
            # Print epoch summary
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            print(f"Epoch Time: {epoch_time:.2f}s")
            
            # Save best model
            # We save the model with highest validation accuracy
            # This is the model that generalizes best to unseen data
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_model_path = os.path.join(
                    checkpoint_dir, 
                    f'best_model_acc_{val_acc:.2f}.pth'
                )
                
                # Save model state
                # Includes:
                # - model.state_dict(): All model weights
                # - optimizer.state_dict(): Optimizer state (for resuming training)
                # - epoch: Current epoch number
                # - metrics: Performance metrics
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                }, self.best_model_path)
                
                print(f"✓ Saved new best model with val_acc: {val_acc:.2f}%")
            
            # Early stopping check
            # If validation accuracy hasn't improved for many epochs,
            # we might be overfitting - consider stopping
            if epoch > 10:
                recent_val_accs = self.history['val_acc'][-5:]
                if max(recent_val_accs) < self.best_val_acc - 5.0:
                    print("\nEarly stopping: Validation accuracy hasn't improved")
                    print(f"Best validation accuracy: {self.best_val_acc:.2f}%")
                    break
        
        # Training complete
        total_time = time.time() - start_time
        print("\n" + "="*60)
        print("Training Complete!")
        print(f"Total Training Time: {total_time/60:.2f} minutes")
        print(f"Best Validation Accuracy: {self.best_val_acc:.2f}%")
        print(f"Best Model Saved: {self.best_model_path}")
        print("="*60)
        
        # Save training history
        history_path = os.path.join(checkpoint_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)
        print(f"Training history saved: {history_path}")
        
        return self.history
    
    def load_checkpoint(self, checkpoint_path):
        """
        Load a saved model checkpoint.
        
        Useful for:
        - Resuming training from a saved state
        - Loading the best model for evaluation
        - Transfer learning (using pre-trained weights)
        
        Args:
            checkpoint_path (str): Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"Checkpoint loaded from: {checkpoint_path}")
        print(f"Epoch: {checkpoint['epoch']}")
        print(f"Validation Accuracy: {checkpoint['val_acc']:.2f}%")
        
        return checkpoint


# Example usage
if __name__ == "__main__":
    print("This module should be imported, not run directly.")
    print("See main.py for complete training pipeline.")
