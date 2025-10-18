import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.5):
        """
        Initialize a 2-layer neural network
        
        Architecture:
        Input Layer -> Hidden Layer -> Output Layer
        """
        self.lr = learning_rate
        
        # Weights: Initialize with small random values
        self.W1 = np.random.randn(input_size, hidden_size) * 0.5
        self.b1 = np.zeros((1, hidden_size))
        
        self.W2 = np.random.randn(hidden_size, output_size) * 0.5
        self.b2 = np.zeros((1, output_size))
        
        print(f"Network initialized with learning rate: {self.lr}")
        print(f"W1 shape: {self.W1.shape} (Input -> Hidden)")
        print(f"W2 shape: {self.W2.shape} (Hidden -> Output)")
    
    def sigmoid(self, z):
        """
        Sigmoid activation function
        Formula: σ(z) = 1 / (1 + e^(-z))
        
        Properties:
        - Outputs between 0 and 1
        - Smooth gradient
        - Good for binary classification
        """
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_derivative(self, a):
        """
        Derivative of sigmoid
        Formula: σ'(z) = σ(z) * (1 - σ(z))
        
        Note: We pass 'a' which is already sigmoid(z)
        This is more efficient!
        """
        return a * (1 - a)
    
    def forward(self, X):
        """
        Forward propagation: Push input through the network
        
        Steps:
        1. Input -> Hidden layer (linear + activation)
        2. Hidden -> Output layer (linear + activation)
        
        We store intermediate values (Z1, A1, Z2, A2) 
        because we'll need them for backprop!
        """
        # Layer 1: Input to Hidden
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.sigmoid(self.Z1)
        
        # Layer 2: Hidden to Output
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.sigmoid(self.Z2)
        
        return self.A2
    
    def backward(self, X, y):
        """
        Backpropagation: Calculate gradients using CHAIN RULE
        
        Goal: Find how much each weight contributed to the error
        Then update weights to reduce error
        
        CHAIN RULE IN ACTION:
        - Start from output error
        - Flow backwards through network
        - Each layer multiplies its local gradient
        """
        m = X.shape[0]  # number of examples
        
        # OUTPUT LAYER GRADIENTS
        # Error at output: how wrong were we?
        # dZ2 = (prediction - actual) * sigmoid'(Z2)
        # But for MSE loss with sigmoid, this simplifies to:
        dZ2 = self.A2 - y
        
        # How much does W2 need to change?
        # dW2 = (1/m) * A1.T · dZ2  (chain rule!)
        dW2 = (1/m) * np.dot(self.A1.T, dZ2)
        
        # How much does b2 need to change?
        db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)
        
        # HIDDEN LAYER GRADIENTS
        # Error flowing back to hidden layer
        # dZ1 = (error from next layer) * sigmoid'(Z1)
        # Chain rule: dZ1 = dZ2 · W2.T * σ'(A1)
        dZ1 = np.dot(dZ2, self.W2.T) * self.sigmoid_derivative(self.A1)
        
        # How much does W1 need to change?
        dW1 = (1/m) * np.dot(X.T, dZ1)
        
        # How much does b1 need to change?
        db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)
        
        # UPDATE WEIGHTS (Gradient Descent)
        # Move weights in opposite direction of gradient
        # New weight = Old weight - learning_rate * gradient
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
    
    def compute_loss(self, y_pred, y_true):
        """
        Binary Cross-Entropy Loss (BCE)
        Formula: L = -(1/m) * Σ[y·log(ŷ) + (1-y)·log(1-ŷ)]
        
        Why BCE for binary classification?
        - Designed for probabilities [0, 1]
        - Heavily penalizes confident wrong predictions
        - Better gradient flow than MSE
        
        Small epsilon added to prevent log(0)
        """
        m = y_true.shape[0]
        epsilon = 1e-15  # prevent log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        loss = -(1/m) * np.sum(
            y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
        )
        return loss
    
    def train(self, X, y, epochs):
        """
        Training loop:
        1. Forward pass (make prediction)
        2. Calculate loss (how wrong?)
        3. Backward pass (calculate gradients)
        4. Update weights
        5. Repeat!
        """
        losses = []
        
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)
            
            # Calculate loss
            loss = self.compute_loss(output, y)
            losses.append(loss)
            
            # Backward pass (update weights)
            self.backward(X, y)
            
            # Print progress
            if epoch % 1000 == 0:
                print(f"Epoch {epoch:5d} | Loss: {loss:.6f}")
        
        return losses
    
    def predict(self, X):
        """Make predictions (forward pass)"""
        output = self.forward(X)
        return (output > 0.5).astype(int)


# XOR Problem Dataset
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([[0],
              [1],
              [1],
              [0]])

print("="*50)
print("TRAINING NEURAL NETWORK ON XOR")
print("="*50)

# Create network
nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1, learning_rate=0.5)

print("\nBEFORE TRAINING:")
print("Input -> Prediction -> Expected")
for i in range(len(X)):
    pred = nn.forward(X[i:i+1])
    print(f"{X[i]} -> {pred[0][0]:.4f} -> {y[i][0]}")

print("\n" + "="*50)
print("TRAINING...")
print("="*50)

# Train the network
losses = nn.train(X, y, epochs=10000)

print("\n" + "="*50)
print("AFTER TRAINING:")
print("="*50)
print("Input -> Prediction -> Expected -> Correct?")
predictions = nn.predict(X)
for i in range(len(X)):
    pred = nn.forward(X[i:i+1])
    is_correct = "✓" if predictions[i][0] == y[i][0] else "✗"
    print(f"{X[i]} -> {pred[0][0]:.4f} -> {y[i][0]} -> {is_correct}")

print(f"\nFinal Loss: {losses[-1]:.6f}")
print(f"Accuracy: {np.mean(predictions == y) * 100:.1f}%")