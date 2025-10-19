"""
CNN Model Architecture for Image Classification

This module defines a Convolutional Neural Network (CNN) for image classification.
It includes detailed mathematical explanations of each layer and operation.

Mathematical Foundations:
1. Convolution: Feature extraction using learnable filters
2. ReLU: Non-linear activation for learning complex patterns
3. Pooling: Spatial dimension reduction and translation invariance
4. Batch Normalization: Stabilizes training by normalizing layer inputs
5. Dropout: Regularization to prevent overfitting
6. Fully Connected: Final classification layers

Key Concepts:
- Feature Maps: Outputs of convolutional layers
- Receptive Field: Area of input that affects a neuron
- Parameter Sharing: Same weights used across spatial locations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    """
    Convolutional Neural Network for CIFAR-10 Classification
    
    Architecture Overview:
    Input (3x32x32) → Conv Block 1 → Conv Block 2 → Conv Block 3 → FC Layers → Output (10)
    
    Each Conv Block contains:
    - Convolutional Layer (feature extraction)
    - Batch Normalization (training stability)
    - ReLU Activation (non-linearity)
    - Max Pooling (dimension reduction)
    - Dropout (regularization)
    """
    
    def __init__(self, num_classes=10, dropout_rate=0.5):
        """
        Initialize the CNN architecture.
        
        Args:
            num_classes (int): Number of output classes (10 for CIFAR-10)
            dropout_rate (float): Probability of dropping neurons (0.5 = 50%)
        
        Mathematical Note:
        - Higher dropout_rate = more regularization but slower learning
        - Typical values: 0.3-0.5 for convolutional layers, 0.5-0.7 for FC layers
        """
        super(CNN, self).__init__()
        
        # ========================================================================
        # CONVOLUTIONAL BLOCK 1
        # Input: 3 channels (RGB), 32x32 pixels
        # Output: 32 channels, 32x32 pixels (with padding)
        # ========================================================================
        
        # First Convolutional Layer
        # Mathematical Operation: Convolution
        # Formula: Output[i,j] = Σ(Input[i+m, j+n] * Kernel[m,n]) + Bias
        # 
        # Parameters:
        # - in_channels=3: RGB input (Red, Green, Blue)
        # - out_channels=32: Number of filters/feature maps to learn
        # - kernel_size=3: 3x3 filter (common choice, captures local patterns)
        # - padding=1: Adds 1 pixel border to maintain spatial dimensions
        # - stride=1: Move filter 1 pixel at a time
        #
        # Why 3x3 kernels?
        # - Captures local patterns (edges, corners, textures)
        # - Smaller than larger kernels (5x5, 7x7) but can stack for larger receptive field
        # - Fewer parameters = less overfitting
        #
        # Receptive Field: Each output pixel sees a 3x3 area of input
        # Number of Parameters: (3 * 3 * 3 + 1) * 32 = 896
        #                       (kernel_h * kernel_w * in_channels + bias) * out_channels
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, 
                              kernel_size=3, padding=1, stride=1)
        
        # Batch Normalization Layer
        # Mathematical Operation: BN(x) = γ * ((x - μ) / √(σ² + ε)) + β
        # Where:
        # - x: Input features
        # - μ: Batch mean
        # - σ²: Batch variance
        # - γ: Learnable scale parameter
        # - β: Learnable shift parameter
        # - ε: Small constant for numerical stability (typically 1e-5)
        #
        # Why Batch Normalization?
        # 1. Reduces internal covariate shift (changing distribution of layer inputs)
        # 2. Allows higher learning rates (faster training)
        # 3. Acts as regularization (reduces need for dropout)
        # 4. Makes network less sensitive to weight initialization
        #
        # Parameters: 2 * num_features = 2 * 32 = 64 (γ and β for each channel)
        self.bn1 = nn.BatchNorm2d(32)
        
        # ========================================================================
        # CONVOLUTIONAL BLOCK 2
        # Input: 32 channels, 16x16 pixels (after pooling from block 1)
        # Output: 64 channels, 16x16 pixels
        # ========================================================================
        
        # Second Convolutional Layer
        # Doubles the number of channels (32 → 64)
        # Why increase channels?
        # - Early layers: Learn simple features (edges, colors)
        # - Deeper layers: Learn complex features (shapes, objects)
        # - More channels = more diverse features
        #
        # Number of Parameters: (3 * 3 * 32 + 1) * 64 = 18,496
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # ========================================================================
        # CONVOLUTIONAL BLOCK 3
        # Input: 64 channels, 8x8 pixels (after pooling from block 2)
        # Output: 128 channels, 8x8 pixels
        # ========================================================================
        
        # Third Convolutional Layer
        # Further increases channels (64 → 128)
        # At this depth, network learns high-level features
        # Examples: object parts, complex textures, patterns
        #
        # Number of Parameters: (3 * 3 * 64 + 1) * 128 = 73,856
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # ========================================================================
        # POOLING LAYER (used after each conv block)
        # ========================================================================
        
        # Max Pooling Layer
        # Mathematical Operation: Output[i,j] = max(Input[2i:2i+2, 2j:2j+2])
        # 
        # Parameters:
        # - kernel_size=2: Takes maximum from 2x2 regions
        # - stride=2: Moves 2 pixels at a time (no overlap)
        #
        # Why Max Pooling?
        # 1. Dimension Reduction: 32x32 → 16x16 → 8x8 → 4x4
        # 2. Translation Invariance: Small shifts in input don't change output much
        # 3. Reduces computation and parameters in later layers
        # 4. Prevents overfitting by reducing spatial information
        # 5. Increases receptive field of subsequent layers
        #
        # No learnable parameters (it's just a max operation)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # ========================================================================
        # DROPOUT LAYERS (Regularization)
        # ========================================================================
        
        # Dropout for Convolutional Layers
        # Mathematical Operation: During training, randomly set neurons to 0
        # Formula: Output = Input * Mask / (1 - p), where Mask ~ Bernoulli(1-p)
        #
        # Why Dropout?
        # 1. Prevents co-adaptation of neurons (neurons relying on specific others)
        # 2. Forces network to learn robust features
        # 3. Acts as ensemble of multiple networks
        # 4. Reduces overfitting
        #
        # During training: Randomly drops 30% of neurons
        # During inference: Uses all neurons (no dropout)
        self.dropout_conv = nn.Dropout2d(p=0.3)
        
        # Dropout for Fully Connected Layers (higher rate)
        # FC layers have more parameters → more prone to overfitting
        # So we use higher dropout rate (50% vs 30%)
        self.dropout_fc = nn.Dropout(p=dropout_rate)
        
        # ========================================================================
        # FULLY CONNECTED LAYERS (Classification Head)
        # After 3 pooling operations: 32→16→8→4
        # Feature map size: 128 channels × 4 × 4 = 2048 features
        # ========================================================================
        
        # First Fully Connected Layer
        # Mathematical Operation: Output = Input @ Weights + Bias
        # Matrix multiplication followed by bias addition
        #
        # Input: 128 * 4 * 4 = 2048 features (flattened from conv layers)
        # Output: 512 neurons
        #
        # Why 512?
        # - Common choice for hidden layer size
        # - Enough capacity to learn complex decision boundaries
        # - Not too large (would overfit and slow training)
        #
        # Number of Parameters: 2048 * 512 + 512 = 1,049,088
        # This is where most parameters are! (Dense connections)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.bn_fc1 = nn.BatchNorm1d(512)  # Batch norm for FC layer
        
        # Second Fully Connected Layer
        # Input: 512 neurons
        # Output: 256 neurons
        # Gradually reduces dimensions toward final output
        #
        # Number of Parameters: 512 * 256 + 256 = 131,328
        self.fc2 = nn.Linear(512, 256)
        self.bn_fc2 = nn.BatchNorm1d(256)
        
        # Output Layer
        # Input: 256 neurons
        # Output: num_classes (10 for CIFAR-10)
        #
        # No activation function here!
        # Why? CrossEntropyLoss applies softmax internally
        #
        # Softmax Formula: P(class_i) = exp(z_i) / Σ(exp(z_j))
        # Converts raw scores (logits) to probabilities
        #
        # Number of Parameters: 256 * 10 + 10 = 2,570
        self.fc3 = nn.Linear(256, num_classes)
        
        # Initialize weights using He initialization
        # Why? ReLU activation benefits from He initialization
        # He initialization: weights ~ N(0, √(2/n_in))
        # Prevents vanishing/exploding gradients
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Initialize network weights using He initialization.
        
        Mathematical Foundation:
        - He initialization: W ~ N(0, σ²), where σ = √(2/n_in)
        - n_in: number of input neurons to the layer
        
        Why He initialization?
        - Designed for ReLU activations
        - Maintains variance of activations across layers
        - Prevents vanishing/exploding gradients
        - Leads to faster convergence
        
        Alternative initializations:
        - Xavier/Glorot: For tanh/sigmoid activations
        - Uniform: Simple but less effective
        - Zero: Bad! Causes symmetry problem
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # He initialization for convolutional layers
                # mode='fan_in': Based on number of input connections
                # nonlinearity='relu': Optimized for ReLU
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # He initialization for fully connected layers
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                # Batch norm: Initialize scale to 1, shift to 0
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Mathematical Flow:
        Input → Conv1 → BN → ReLU → Pool → Dropout →
        Conv2 → BN → ReLU → Pool → Dropout →
        Conv3 → BN → ReLU → Pool → Dropout →
        Flatten → FC1 → BN → ReLU → Dropout →
        FC2 → BN → ReLU → Dropout →
        FC3 → Output (logits)
        
        Args:
            x (torch.Tensor): Input batch of images [batch_size, 3, 32, 32]
        
        Returns:
            torch.Tensor: Output logits [batch_size, num_classes]
        
        Shape Transformations:
        [B, 3, 32, 32]   Input
        [B, 32, 32, 32]  After Conv1
        [B, 32, 16, 16]  After Pool1
        [B, 64, 16, 16]  After Conv2
        [B, 64, 8, 8]    After Pool2
        [B, 128, 8, 8]   After Conv3
        [B, 128, 4, 4]   After Pool3
        [B, 2048]        After Flatten
        [B, 512]         After FC1
        [B, 256]         After FC2
        [B, 10]          After FC3 (Output)
        
        Where B = batch_size
        """
        
        # ========================================================================
        # CONVOLUTIONAL BLOCK 1
        # ========================================================================
        
        # Convolution: Extract low-level features (edges, colors, simple textures)
        x = self.conv1(x)  # [B, 3, 32, 32] → [B, 32, 32, 32]
        
        # Batch Normalization: Normalize activations
        # Helps gradients flow better during backpropagation
        x = self.bn1(x)
        
        # ReLU Activation: Introduce non-linearity
        # Formula: ReLU(x) = max(0, x)
        # Why ReLU?
        # 1. Computationally efficient (just max operation)
        # 2. Doesn't saturate for positive values (unlike sigmoid/tanh)
        # 3. Sparse activation (many zeros → faster computation)
        # 4. Helps with vanishing gradient problem
        x = F.relu(x)
        
        # Max Pooling: Reduce spatial dimensions by half
        # Increases receptive field and reduces computation
        x = self.pool(x)  # [B, 32, 32, 32] → [B, 32, 16, 16]
        
        # Dropout: Randomly zero out 30% of activations (only during training)
        # Prevents overfitting by making network more robust
        x = self.dropout_conv(x)
        
        # ========================================================================
        # CONVOLUTIONAL BLOCK 2
        # ========================================================================
        
        # Extract mid-level features (shapes, patterns, object parts)
        x = self.conv2(x)  # [B, 32, 16, 16] → [B, 64, 16, 16]
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)   # [B, 64, 16, 16] → [B, 64, 8, 8]
        x = self.dropout_conv(x)
        
        # ========================================================================
        # CONVOLUTIONAL BLOCK 3
        # ========================================================================
        
        # Extract high-level features (object representations, semantic features)
        x = self.conv3(x)  # [B, 64, 8, 8] → [B, 128, 8, 8]
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)   # [B, 128, 8, 8] → [B, 128, 4, 4]
        x = self.dropout_conv(x)
        
        # ========================================================================
        # FLATTEN LAYER
        # ========================================================================
        
        # Flatten: Convert 3D feature maps to 1D vector
        # [B, 128, 4, 4] → [B, 128*4*4] = [B, 2048]
        # Necessary for fully connected layers which expect 1D input
        x = x.view(x.size(0), -1)  # -1 infers the dimension automatically
        
        # ========================================================================
        # FULLY CONNECTED LAYERS (Classification Head)
        # ========================================================================
        
        # First FC layer: Learn complex combinations of extracted features
        x = self.fc1(x)       # [B, 2048] → [B, 512]
        x = self.bn_fc1(x)
        x = F.relu(x)
        x = self.dropout_fc(x)  # Higher dropout (50%) for FC layers
        
        # Second FC layer: Further refine feature representations
        x = self.fc2(x)       # [B, 512] → [B, 256]
        x = self.bn_fc2(x)
        x = F.relu(x)
        x = self.dropout_fc(x)
        
        # Output layer: Produce class scores (logits)
        # No activation here! CrossEntropyLoss handles softmax
        x = self.fc3(x)       # [B, 256] → [B, 10]
        
        return x  # Returns raw scores (logits) for each class
    
    def count_parameters(self):
        """
        Count total trainable parameters in the model.
        
        Why important?
        - More parameters = more capacity but risk of overfitting
        - Fewer parameters = less capacity but better generalization
        - Modern CNNs: millions to billions of parameters
        
        Returns:
            int: Total number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_feature_maps(self, x, layer_name):
        """
        Extract feature maps from a specific layer for visualization.
        
        Useful for understanding what the network learns:
        - Early layers: Edges, colors, simple patterns
        - Middle layers: Shapes, textures, object parts
        - Deep layers: High-level object representations
        
        Args:
            x (torch.Tensor): Input image
            layer_name (str): Name of layer to extract ('conv1', 'conv2', 'conv3')
        
        Returns:
            torch.Tensor: Feature maps from specified layer
        """
        if layer_name == 'conv1':
            x = self.conv1(x)
            x = self.bn1(x)
            x = F.relu(x)
            return x
        elif layer_name == 'conv2':
            x = self.conv1(x)
            x = self.bn1(x)
            x = F.relu(x)
            x = self.pool(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = F.relu(x)
            return x
        elif layer_name == 'conv3':
            x = self.conv1(x)
            x = self.bn1(x)
            x = F.relu(x)
            x = self.pool(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = F.relu(x)
            x = self.pool(x)
            x = self.conv3(x)
            x = self.bn3(x)
            x = F.relu(x)
            return x
        else:
            raise ValueError(f"Unknown layer: {layer_name}")


# Example usage and model summary
if __name__ == "__main__":
    print("="*60)
    print("Testing CNN Model Architecture")
    print("="*60)
    
    # Create model instance
    model = CNN(num_classes=10, dropout_rate=0.5)
    
    # Print model architecture
    print("\nModel Architecture:")
    print(model)
    
    # Count parameters
    total_params = model.count_parameters()
    print(f"\nTotal Trainable Parameters: {total_params:,}")
    
    # Calculate model size in MB
    param_size_mb = total_params * 4 / (1024 ** 2)  # 4 bytes per parameter (float32)
    print(f"Model Size: {param_size_mb:.2f} MB")
    
    # Test forward pass with dummy input
    dummy_input = torch.randn(4, 3, 32, 32)  # Batch of 4 images
    print(f"\nInput shape: {dummy_input.shape}")
    
    # Forward pass
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    print(f"Output (logits) for first image: {output[0]}")
    
    # Apply softmax to get probabilities
    probabilities = F.softmax(output[0], dim=0)
    print(f"Probabilities (after softmax): {probabilities}")
    print(f"Sum of probabilities: {probabilities.sum():.4f}")  # Should be 1.0
    
    # Get predicted class
    predicted_class = output[0].argmax().item()
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {probabilities[predicted_class]:.4f}")
    
    print("\n" + "="*60)
    print("Model architecture test completed successfully!")
    print("="*60)
