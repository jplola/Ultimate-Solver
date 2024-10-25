import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Optional


class Layer(ABC):
    """Abstract base class for all layers"""

    @abstractmethod
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Forward pass"""
        pass

    @abstractmethod
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward pass"""
        pass


import numpy as np
from typing import Tuple, Optional


class BatchNorm2D(Layer):
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        """
        Parameters:
        - num_features: Number of channels/features
        - eps: Small constant for numerical stability
        - momentum: Momentum for running statistics
        """
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # Learnable parameters
        self.gamma = np.ones(num_features)  # Scale parameter
        self.beta = np.zeros(num_features)  # Shift parameter

        # Running statistics for inference
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

        # Gradients
        self.gamma_grad = np.zeros_like(self.gamma)
        self.beta_grad = np.zeros_like(self.beta)

        self.cache = {}
        self.training = True

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Forward pass for batch normalization

        Parameters:
        - inputs: Shape (batch_size, height, width, channels)

        Returns:
        - outputs: Shape (batch_size, height, width, channels)
        """
        # Cache inputs for backprop
        self.cache['inputs'] = inputs

        # Reshape inputs for normalization
        batch_size, height, width, channels = inputs.shape
        inputs_reshaped = inputs.reshape(-1, channels)

        if self.training:
            # Calculate mean and variance for current batch
            batch_mean = np.mean(inputs_reshaped, axis=0)
            batch_var = np.var(inputs_reshaped, axis=0)

            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
        else:
            # Use running statistics during inference
            batch_mean = self.running_mean
            batch_var = self.running_var

        # Normalize
        x_normalized = (inputs_reshaped - batch_mean) / np.sqrt(batch_var + self.eps)

        # Scale and shift
        out = self.gamma * x_normalized + self.beta

        # Cache variables for backward pass
        self.cache.update({
            'batch_mean': batch_mean,
            'batch_var': batch_var,
            'x_normalized': x_normalized,
            'sqrt_var': np.sqrt(batch_var + self.eps)
        })

        # Reshape back to original shape
        return out.reshape(batch_size, height, width, channels)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """
        Backward pass for batch normalization

        Parameters:
        - grad: Gradient from next layer

        Returns:
        - input_grad: Gradient with respect to input
        """
        inputs = self.cache['inputs']
        batch_mean = self.cache['batch_mean']
        batch_var = self.cache['batch_var']
        x_normalized = self.cache['x_normalized']
        sqrt_var = self.cache['sqrt_var']

        batch_size, height, width, channels = inputs.shape
        grad_reshaped = grad.reshape(-1, channels)
        inputs_reshaped = inputs.reshape(-1, channels)
        N = batch_size * height * width

        # Gradient with respect to beta
        self.beta_grad = np.sum(grad_reshaped, axis=0)

        # Gradient with respect to gamma
        self.gamma_grad = np.sum(grad_reshaped * x_normalized, axis=0)

        # Gradient with respect to x_normalized
        dx_normalized = grad_reshaped * self.gamma

        # Gradient with respect to variance
        dvar = np.sum(dx_normalized * (inputs_reshaped - batch_mean) * -0.5
                      * (batch_var + self.eps) ** (-1.5), axis=0)

        # Gradient with respect to mean
        dmean = np.sum(dx_normalized * -1 / sqrt_var, axis=0) + \
                dvar * np.mean(-2 * (inputs_reshaped - batch_mean), axis=0)

        # Gradient with respect to input
        input_grad = dx_normalized / sqrt_var + \
                     dvar * 2 * (inputs_reshaped - batch_mean) / N + \
                     dmean / N

        return input_grad.reshape(batch_size, height, width, channels)


class Conv2D(Layer):
    def __init__(self, filters: int, kernel_size: Tuple[int, int],
                 stride: int = 1, padding: int = 0, use_batchnorm: bool = True):
        """
        Parameters:
        - filters: Number of filters/kernels
        - kernel_size: (height, width) of each filter
        - stride: Stride for the convolution
        - padding: Zero-padding size
        - use_batchnorm: Whether to use batch normalization
        """
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_batchnorm = use_batchnorm

        # To be initialized when first input is received
        self.weights = None
        self.bias = None if use_batchnorm else None  # No bias needed with BatchNorm
        self.batchnorm = BatchNorm2D(filters) if use_batchnorm else None
        self.cache = {}

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Forward pass with optional batch normalization"""
        # Initialize weights if first forward pass
        if self.weights is None:
            self._initialize_weights(inputs.shape[-1])

        # Standard convolution operation (previous implementation)
        conv_output = super().forward(inputs)

        # Apply batch normalization if enabled
        if self.use_batchnorm:
            return self.batchnorm.forward(conv_output)
        return conv_output

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward pass with optional batch normalization"""
        if self.use_batchnorm:
            grad = self.batchnorm.backward(grad)

        # Standard convolution backward pass (previous implementation)
        return super().backward(grad)

    def train(self):
        """Set layer to training mode"""
        if self.batchnorm:
            self.batchnorm.training = True

    def eval(self):
        """Set layer to evaluation mode"""
        if self.batchnorm:
            self.batchnorm.training = False


# Example usage

class ReLU(Layer):
    def __init__(self):
        self.cache = {}

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        TODO: Implement ReLU activation function
        f(x) = max(0, x)
        """
        raise NotImplementedError

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """
        TODO: Implement ReLU backward pass
        Gradient is 1 where input was > 0, else 0
        """
        raise NotImplementedError


class MaxPool2D(Layer):
    def __init__(self, pool_size: Tuple[int, int], stride: Optional[int] = None):
        """
        Parameters:
        - pool_size: (height, width) of pooling window
        - stride: Stride for pooling (defaults to pool_size if None)
        """
        self.pool_size = pool_size
        self.stride = stride if stride is not None else pool_size[0]
        self.cache = {}

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        TODO: Implement max pooling forward pass
        1. For each pooling window:
            - Find maximum value
            - Store indices for backward pass
        2. Return pooled output
        """
        raise NotImplementedError

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """
        TODO: Implement max pooling backward pass
        1. Distribute gradients only to maximum elements from forward pass
        2. All other elements receive zero gradient
        """
        raise NotImplementedError


class Flatten(Layer):
    def __init__(self):
        self.cache = {}

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        TODO: Implement flatten operation
        Convert multi-dimensional input to 2D: (batch_size, features)
        """
        raise NotImplementedError

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """
        TODO: Implement flatten backward pass
        Reshape gradients back to input shape
        """
        raise NotImplementedError


class Dense(Layer):
    def __init__(self, units: int):
        """
        Parameters:
        - units: Number of output neurons
        """
        self.units = units
        self.weights: Optional[np.ndarray] = None
        self.bias: Optional[np.ndarray] = None
        self.cache = {}

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        TODO: Implement dense layer forward pass
        1. Initialize weights if needed (He initialization)
        2. Compute output = inputs @ weights + bias
        3. Cache values for backward pass
        """
        raise NotImplementedError

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """
        TODO: Implement dense layer backward pass
        1. Compute gradients with respect to inputs
        2. Compute gradients with respect to weights
        3. Compute gradients with respect to bias
        4. Store weight and bias gradients
        """
        raise NotImplementedError


class Softmax(Layer):
    def __init__(self):
        self.cache = {}

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        TODO: Implement softmax activation
        1. Compute exp(x - max(x)) for numerical stability
        2. Normalize by sum
        """
        raise NotImplementedError

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """
        TODO: Implement softmax backward pass
        Gradient with respect to inputs considering cross-entropy loss
        """
        raise NotImplementedError


class CNN:
    def __init__(self):
        """
        Initialize CNN architecture
        Example architecture: Conv -> ReLU -> MaxPool -> Flatten -> Dense -> Softmax
        """
        self.layers = [
            Conv2D(filters=16, kernel_size=(3, 3), stride=1, padding=1),
            ReLU(),
            MaxPool2D(pool_size=(2, 2)),
            Flatten(),
            Dense(units=10),
            Softmax()
        ]

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through all layers"""
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad: np.ndarray) -> None:
        """Backward pass through all layers"""
        for layer in reversed(self.layers):
            grad = layer.backward(grad)


# Example usage
def create_example_cnn():
    # Create sample data (batch_size, height, width, channels)
    x = np.random.randn(32, 28, 28, 1)

    # Initialize CNN
    cnn = CNN()

    # Forward pass
    output = cnn.forward(x)

    # Assuming we have labels y and loss gradient
    grad = np.random.randn(*output.shape)  # This would normally come from loss function

    # Backward pass
    cnn.backward(grad)

    return output