import numpy as np 

np.random.seed(42)

class NeuralNetwork:
    def __init__ (self, hidden_size, input_size, output_size):
        """
        Initialize a 2 lsyer neural Network

        Architecutre: Input Layer - Hidden Layer - Output Layer
        """

        #wights are being initialize with small random values
        #small because large weight can cuase exploding gradients

        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))

        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

        print(f"Network Initialized")
        print(f"W1 shape: {self.W1.shape} (Input -> Hidden)")
        print(f"W2 shape: {self.W2.shape} (Hidden -> Output)")


    def sigmoid(self, z):
        """
        sigmoid activation function
        formula: σ(z) = 1 / (1 + e^(-z))

        Properties:
        - Outputs between 0 to 1
        - Smooth gradient
        - Used for binrary  classification
        """
        return 1 / (1 + np.exp(-z))

    def forward(self, X):
        """
        Forward prop: push input through the network

        Steps:
        1. Input -> Hidden layer (linear transformation + activation)
        2. Hidden -> Output layer (linear transformation + activation) 
        """

        #Layer 1: Input to hidden
        #Z1 = X * W1 + b1
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.sigmoid(self.Z1)

        #Layer 2: hidden to output
        #Z2 = A1 * W2 + b2
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.sigmoid(self.Z2)

        return self.A2

    def predict (self, X):
        """make predictions"""
        output = self.forward(x)
        return (output > 0.5).astype(int)


# XOR problem dataset
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0],[1],[1],[0]])

nn = NeuralNetwork(input_size= 2, hidden_size=4, output_size=1)

print("\n" + "="*50)
print("FORWARD PASS TEST")
print("="*50)

print("\nInput data (X):")
print(X)

print("\nExpected output (y):")
print(y.T)

predictions = nn.forward(X)
print("\nNetwork predictions (untrained, will be random):")
print(predictions.T)

print("\n" + "="*50)
print("Notice: Predictions are random because we haven't")
print("trained the network yet! Next: Backpropagation!")
print("="*50)
