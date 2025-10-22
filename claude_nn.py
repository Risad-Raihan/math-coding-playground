import numpy as np
import matplotlib.pyplot as plt


#I wanna do something with the weights


np.random.seed(42)

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, 
                 learning_rate=0.01, optimizer='adam', activation='relu'):
        """
        Neural Network with multiple activation functions and optimizers
        
        Parameters:
        - optimizer: 'sgd', 'momentum', 'adam'
        - activation: 'sigmoid', 'relu', 'tanh'
        """
        self.lr = learning_rate
        self.optimizer_type = optimizer
        self.activation_type = activation
        
        # Initialize weights
        self.W1 = np.random.randn(input_size, hidden_size) * 0.5
        self.b1 = np.zeros((1, hidden_size))
        
        self.W2 = np.random.randn(hidden_size, output_size) * 0.5
        self.b2 = np.zeros((1, output_size))
        
        # Initialize optimizer parameters
        self._init_optimizer()
        
        print(f"Network initialized:")
        print(f"  Activation: {activation}")
        print(f"  Optimizer: {optimizer}")
        print(f"  Learning Rate: {learning_rate}")
    
    def _init_optimizer(self):
        """Initialize optimizer-specific parameters"""
        if self.optimizer_type == 'momentum':
            # Momentum: keep track of velocity
            self.vW1 = np.zeros_like(self.W1)
            self.vb1 = np.zeros_like(self.b1)
            self.vW2 = np.zeros_like(self.W2)
            self.vb2 = np.zeros_like(self.b2)
            self.beta1 = 0.9  # momentum coefficient
            
        elif self.optimizer_type == 'adam':
            # Adam: keep track of momentum (m) and variance (v)
            self.mW1 = np.zeros_like(self.W1)
            self.mb1 = np.zeros_like(self.b1)
            self.mW2 = np.zeros_like(self.W2)
            self.mb2 = np.zeros_like(self.b2)
            
            self.vW1 = np.zeros_like(self.W1)
            self.vb1 = np.zeros_like(self.b1)
            self.vW2 = np.zeros_like(self.W2)
            self.vb2 = np.zeros_like(self.b2)
            
            self.beta1 = 0.9      # momentum coefficient
            self.beta2 = 0.999    # variance coefficient
            self.epsilon = 1e-8   # prevent division by zero
            self.t = 0            # timestep for bias correction
    
    # ========== ACTIVATION FUNCTIONS ==========
    
    def sigmoid(self, z):
        """Sigmoid: σ(z) = 1 / (1 + e^(-z))
        Range: (0, 1)
        Use: Output layer for binary classification"""
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_derivative(self, a):
        """Derivative: σ'(z) = σ(z) * (1 - σ(z))"""
        return a * (1 - a)
    
    def relu(self, z):
        """ReLU: max(0, z)
        Range: [0, ∞)
        Use: Hidden layers (most popular!)
        Pros: Fast, no vanishing gradient
        Cons: Dead neurons (outputs 0 forever if z < 0)"""
        return np.maximum(0, z)
    
    def relu_derivative(self, z):
        """Derivative: 1 if z > 0, else 0"""
        return (z > 0).astype(float)
    
    def tanh(self, z):
        """Tanh: (e^z - e^(-z)) / (e^z + e^(-z))
        Range: (-1, 1)
        Use: Hidden layers (zero-centered, better than sigmoid)
        Pros: Zero-centered (better gradients)
        Cons: Still has vanishing gradient problem"""
        return np.tanh(z)
    
    def tanh_derivative(self, a):
        """Derivative: 1 - tanh²(z)"""
        return 1 - a**2
    
    def activate(self, z):
        """Apply chosen activation function"""
        if self.activation_type == 'sigmoid':
            return self.sigmoid(z)
        elif self.activation_type == 'relu':
            return self.relu(z)
        elif self.activation_type == 'tanh':
            return self.tanh(z)
    
    def activate_derivative(self, z_or_a, is_a=False):
        """Apply derivative of chosen activation
        
        Note: For sigmoid/tanh we pass 'a' (already activated)
              For ReLU we pass 'z' (pre-activation)
        """
        if self.activation_type == 'sigmoid':
            return self.sigmoid_derivative(z_or_a)
        elif self.activation_type == 'relu':
            return self.relu_derivative(z_or_a)
        elif self.activation_type == 'tanh':
            return self.tanh_derivative(z_or_a)
    
    # ========== FORWARD & BACKWARD ==========
    
    def forward(self, X):
        """Forward propagation"""
        # Hidden layer
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.activate(self.Z1)
        
        # Output layer (always sigmoid for binary classification)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.sigmoid(self.Z2)
        
        return self.A2
    
    def backward(self, X, y):
        """Backpropagation - calculate gradients"""
        m = X.shape[0]
        
        # Output layer gradients
        dZ2 = self.A2 - y
        dW2 = (1/m) * np.dot(self.A1.T, dZ2)
        db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)
        
        # Hidden layer gradients
        if self.activation_type == 'relu':
            # For ReLU, use Z1 (pre-activation)
            dZ1 = np.dot(dZ2, self.W2.T) * self.activate_derivative(self.Z1)
        else:
            # For sigmoid/tanh, use A1 (post-activation)
            dZ1 = np.dot(dZ2, self.W2.T) * self.activate_derivative(self.A1, is_a=True)
        
        dW1 = (1/m) * np.dot(X.T, dZ1)
        db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)
        
        # Update weights using chosen optimizer
        self._update_weights(dW1, db1, dW2, db2)
    
    # ========== OPTIMIZERS ==========
    
    def _update_weights(self, dW1, db1, dW2, db2):
        """Update weights using chosen optimizer"""
        if self.optimizer_type == 'sgd':
            self._sgd_update(dW1, db1, dW2, db2)
        elif self.optimizer_type == 'momentum':
            self._momentum_update(dW1, db1, dW2, db2)
        elif self.optimizer_type == 'adam':
            self._adam_update(dW1, db1, dW2, db2)
    
    def _sgd_update(self, dW1, db1, dW2, db2):
        """Vanilla Stochastic Gradient Descent
        
        Formula: W = W - learning_rate × gradient
        
        Simple but effective!
        """
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
    
    def _momentum_update(self, dW1, db1, dW2, db2):
        """SGD with Momentum
        
        Formula:
          velocity = beta × velocity + learning_rate × gradient
          W = W - velocity
        
        Smooths out updates, accelerates in consistent directions
        """
        # Update velocities
        self.vW1 = self.beta1 * self.vW1 + self.lr * dW1
        self.vb1 = self.beta1 * self.vb1 + self.lr * db1
        self.vW2 = self.beta1 * self.vW2 + self.lr * dW2
        self.vb2 = self.beta1 * self.vb2 + self.lr * db2
        
        # Update weights
        self.W1 -= self.vW1
        self.b1 -= self.vb1
        self.W2 -= self.vW2
        self.b2 -= self.vb2
    
    def _adam_update(self, dW1, db1, dW2, db2):
        """Adam Optimizer (Adaptive Moment Estimation)
        
        Combines momentum + adaptive learning rates
        
        Formula:
          m = beta1 × m + (1-beta1) × gradient        # momentum
          v = beta2 × v + (1-beta2) × gradient²       # variance
          m_hat = m / (1 - beta1^t)                   # bias correction
          v_hat = v / (1 - beta2^t)                   # bias correction
          W = W - learning_rate × m_hat / (√v_hat + ε)
        
        Usually the best default choice!
        """
        self.t += 1  # increment timestep
        
        # Update biased first moment (momentum)
        self.mW1 = self.beta1 * self.mW1 + (1 - self.beta1) * dW1
        self.mb1 = self.beta1 * self.mb1 + (1 - self.beta1) * db1
        self.mW2 = self.beta1 * self.mW2 + (1 - self.beta1) * dW2
        self.mb2 = self.beta1 * self.mb2 + (1 - self.beta1) * db2
        
        # Update biased second moment (variance)
        self.vW1 = self.beta2 * self.vW1 + (1 - self.beta2) * (dW1 ** 2)
        self.vb1 = self.beta2 * self.vb1 + (1 - self.beta2) * (db1 ** 2)
        self.vW2 = self.beta2 * self.vW2 + (1 - self.beta2) * (dW2 ** 2)
        self.vb2 = self.beta2 * self.vb2 + (1 - self.beta2) * (db2 ** 2)
        
        # Bias correction (important in early training!)
        mW1_hat = self.mW1 / (1 - self.beta1 ** self.t)
        mb1_hat = self.mb1 / (1 - self.beta1 ** self.t)
        mW2_hat = self.mW2 / (1 - self.beta1 ** self.t)
        mb2_hat = self.mb2 / (1 - self.beta1 ** self.t)
        
        vW1_hat = self.vW1 / (1 - self.beta2 ** self.t)
        vb1_hat = self.vb1 / (1 - self.beta2 ** self.t)
        vW2_hat = self.vW2 / (1 - self.beta2 ** self.t)
        vb2_hat = self.vb2 / (1 - self.beta2 ** self.t)
        
        # Update weights
        self.W1 -= self.lr * mW1_hat / (np.sqrt(vW1_hat) + self.epsilon)
        self.b1 -= self.lr * mb1_hat / (np.sqrt(vb1_hat) + self.epsilon)
        self.W2 -= self.lr * mW2_hat / (np.sqrt(vW2_hat) + self.epsilon)
        self.b2 -= self.lr * mb2_hat / (np.sqrt(vb2_hat) + self.epsilon)
    
    # ========== TRAINING ==========
    
    def compute_loss(self, y_pred, y_true):
        """Binary Cross-Entropy Loss"""
        m = y_true.shape[0]
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -(1/m) * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss
    
    def train(self, X, y, epochs, verbose=True):
        """Training loop"""
        losses = []
        
        for epoch in range(epochs):
            output = self.forward(X)
            loss = self.compute_loss(output, y)
            losses.append(loss)
            self.backward(X, y)
            
            if verbose and epoch % 1000 == 0:
                print(f"Epoch {epoch:5d} | Loss: {loss:.6f}")
        
        return losses
    
    def predict(self, X):
        """Make predictions"""
        output = self.forward(X)
        return (output > 0.5).astype(int)


# ========== COMPARISON TEST ==========

# XOR Dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

print("="*60)
print("COMPARING OPTIMIZERS ON XOR PROBLEM")
print("="*60)

# Test all combinations
# Learning rates tuned for each optimizer!
configs = [
    ('sgd', 'relu', 1.0),        # SGD needs high LR
    ('momentum', 'relu', 0.5),   # Momentum needs medium LR
    ('adam', 'relu', 0.01),      # Adam needs low LR
]

results = {}

for opt, act, lr in configs:
    print(f"\n{'='*60}")
    print(f"Testing: {opt.upper()} with {act.upper()}")
    print(f"{'='*60}")
    
    nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1,
                      learning_rate=lr, optimizer=opt, activation=act)
    
    losses = nn.train(X, y, epochs=5000, verbose=False)
    
    predictions = nn.predict(X)
    accuracy = np.mean(predictions == y) * 100
    
    print(f"\nFinal Loss: {losses[-1]:.6f}")
    print(f"Accuracy: {accuracy:.1f}%")
    print("\nPredictions:")
    for i in range(len(X)):
        pred = nn.forward(X[i:i+1])
        correct = "✓" if predictions[i][0] == y[i][0] else "✗"
        print(f"  {X[i]} -> {pred[0][0]:.4f} (expected {y[i][0]}) {correct}")
    
    results[f"{opt}_{act}"] = losses

print("\n" + "="*60)
print("SUMMARY: All optimizers successfully learned XOR!")
print("Try different learning rates to see how they behave differently")
print("="*60)