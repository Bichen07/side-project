import numpy as np
import matplotlib.pyplot as plt

#  activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return np.multiply(x, 1.0 - x) 

def RelU(x):
    return np.where(x > 0, x, 0)
    # return np.clip(x, 0, 5) # clip the value to 0~100

def RelU_derivative(x):
    return np.where(x > 0, 1, 0)

def compute_loss(z, z_hat): # MSE
    n = z.shape[0]
    loss = np.sum((z - z_hat)**2)/n
    grad = 2*(z - z_hat)/n
    return loss, grad

# Generate data functions
def generate_linear(n=100):
    pts = np.random.uniform(0, 1, (n, 2)) # 100 points in 2D, 100*2
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0], pt[1]])
        distance = (pt[0] - pt[1]) / 1.414
        if pt[0] > pt[1]:
            labels.append(1)
        else:
            labels.append(0)
    return np.array(inputs), np.array(labels).reshape(n, 1)

def generate_XOR_easy():
    inputs = []
    labels = []
    for i in range(11):
        inputs.append([0.1 * i, 0.1 * i])
        labels.append(0)
        if 0.1 * i == 0.5:
            continue
        inputs.append([0.1 * i, 1.0 - 0.1 * i])
        labels.append(1)
    return np.array(inputs), np.array(labels).reshape(21, 1)

def display_result(losses, x_test, z_test, z_hat_test, optimizer):
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.suptitle("Learning Curve", fontsize=18, x=0.5, ha='center')
    plt.figtext(0.65, 0.95, f"Optimizer: {optimizer}", fontsize=12, ha='left')
    plt.subplots_adjust(top=0.94) # adjust the space between the title and the subplots
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.plot(losses)

    plt.subplots_adjust(hspace=0.4) # adjust the space between two subplots

    plt.subplot(2, 2, 3)
    plt.title("Ground true", fontsize=18)
    for i in range(z_test.size):
        if z_hat_test[i] == 1:
            plt.plot(x_test[i][0], x_test[i][1], marker='o', color='hotpink')
        else:
            plt.plot(x_test[i][0], x_test[i][1], marker='o', color='cornflowerblue')

    plt.subplot(2, 2, 4)
    accuracy = np.mean(z_test == z_hat_test)  
    plt.text(0, 0, f'Accuracy: {accuracy:.0%}', fontsize=15, bbox=dict(facecolor='white', alpha=0.5))
    plt.title('Predict result', fontsize=18)
    for i in range(x_test.shape[0]):
        if z_test[i] == 1:
            plt.plot(x_test[i][0], x_test[i][1], marker='o', color='hotpink')
        else:
            plt.plot(x_test[i][0], x_test[i][1], marker='o', color='cornflowerblue')
        if z_test[i] != z_hat_test[i]:
            # plt.plot(x_test[i][0], x_test[i][1], marker='o', color='#FFA500') #  wrong prediction with orange circle
            plt.plot(x_test[i][0], x_test[i][1], 'kx') # black cross
    plt.show()

class concolutional_layer():
    def __init__(self):
        pass

class Layer():
    def __init__(self, input_size, output_size, lr, activate, optimizer, batch_size, bias):
        self.input_size = input_size
        self.output_size = output_size
        self.lr = lr
        self.activate = activate
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.bias = bias
        self.W = np.random.normal(size=(input_size, output_size))

        # for optimizer "momentum"
        self.beta = 0.9
        self.velocity = np.zeros_like(self.W)
        self.velocity_bias = np.random.randn()/100 # Random initialization

        # Initialize accumulators for Adagrad and RMSProp
        self.accumulated_grad_squared = np.zeros_like(self.W)  # For Adagrad and RMSProp
        self.accumulated_bias_grad_squared = np.zeros_like(self.bias, dtype=np.float64) # For Adagrad and RMSProp
        self.epsilon = 1e-8  # Small value to prevent division by zero
        self.rmsP = 0.9  # Typically used for RMSProp

        # Initialize accumulators for Adam
        self.m = np.zeros_like(self.W)  # first moment
        self.v = np.zeros_like(self.W)  # second moment
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.t = 0

    def forward(self, input):
        self.input = input
        if(self.activate == "sigmoid"):
            self.output = sigmoid(np.dot(input, self.W) + self.bias)
        elif(self.activate == "RelU"):
            self.output = RelU(np.dot(input, self.W) + self.bias)
        else: # no activation
            self.output = np.dot(input, self.W) + self.bias
        return self.output

    def backward(self, grad):
        if(self.activate == "sigmoid"):
            grad = np.multiply(grad, sigmoid_derivative(self.output))
        elif(self.activate == "RelU"):
            grad = np.multiply(grad, RelU_derivative(self.output))
        else:
            grad = grad

        # optimizer = "SGD", "Mini_Batch_GD", "GD", "Adam", "RMSprop", "momentum"
        if(self.optimizer in {"SGD", "Mini_Batch_GD", "GD"}):
            self.W -= np.dot(self.input.T, grad) * self.lr
            self.bias -= np.sum(grad) * self.lr
        
        elif(self.optimizer == "momentum"):
            self.velocity = self.beta * self.velocity + self.lr * np.dot(self.input.T, grad)
            self.velocity_bias = self.beta * self.velocity_bias + self.lr * np.sum(grad)
            self.W -= self.velocity
            self.bias -= self.velocity_bias

        elif(self.optimizer == "AdaGrad"):
            self.accumulated_grad_squared += np.dot(self.input.T, grad)**2
            self.accumulated_bias_grad_squared += np.sum(grad) ** 2
            self.W -= self.lr * np.dot(self.input.T, grad) / (np.sqrt(self.accumulated_grad_squared) + self.epsilon)
            self.bias -= self.lr * np.sum(grad) / (np.sqrt(self.accumulated_bias_grad_squared) + self.epsilon)
        
        elif(self.optimizer == "RMSprop"):
            self.accumulated_grad_squared = self.rmsP * self.accumulated_grad_squared + (1 - self.rmsP) * np.dot(self.input.T, grad)**2
            self.accumulated_bias_grad_squared = self.rmsP * self.accumulated_bias_grad_squared + (1 - self.rmsP) * (np.sum(grad) ** 2)
            self.W -= self.lr * np.dot(self.input.T, grad) / (np.sqrt(self.accumulated_grad_squared) + self.epsilon)
            self.bias -= self.lr * np.sum(grad) / (np.sqrt(self.accumulated_bias_grad_squared) + self.epsilon)
        
        elif(self.optimizer == "Adam"): # combination of momentum and RMSprop
            self.t += 1
            g = np.dot(self.input.T, grad)
            self.m = self.beta1 * self.m + (1 - self.beta1) * g
            self.v = self.beta2 * self.v + (1 - self.beta2) * (g**2)
            m_hat = self.m / (1 - self.beta1**self.t)
            v_hat = self.v / (1 - self.beta2**self.t)
            self.W -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        else: # no such optimizer
            raise NameError("No such optimizer")
        return np.dot(grad, self.W.T)

class Model():
    def __init__(self, neuron_of_hidden, lr = 0.001, activate = "sigmoid", optimizer = "Mini_Batch_GD", batch_size = 10, bias = 0):
        self.neuron_of_hidden = neuron_of_hidden
        neuron_of_input = 2
        self.lr = lr
        self.activate = activate
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.losses = []
        self.layer_1 = Layer(neuron_of_input, neuron_of_hidden, lr, activate, optimizer, batch_size, bias)
        self.layer_2 = Layer(neuron_of_hidden, neuron_of_hidden, lr, activate, optimizer, batch_size, bias)
        self.layer_output = Layer(neuron_of_hidden, 1, lr, activate, optimizer, batch_size, bias)

    def test(self, input):
        z = self.layer_output.forward(self.layer_2.forward(self.layer_1.forward(input)))
        return z        

    def train(self, x, z_hat, epoch, show_epoch):
        n = x.shape[0]

        if(self.optimizer == "SGD"):
            for i in range(epoch):
                indices = np.random.permutation(n) # Randomly permute a sequence.
                x_shuffled = x[indices]
                z_hat_shuffled = z_hat[indices]
                for j in range(n):
                    z = self.layer_output.forward(self.layer_2.forward(self.layer_1.forward(x[j:j+1])))
                    loss, grad = compute_loss(z, z_hat[j:j+1])
                    self.layer_1.backward(self.layer_2.backward(self.layer_output.backward(grad)))
                self.losses.append(loss)
                if(i % show_epoch == 0):
                    print(f"epoch {i} loss : {loss:.8f}")
        elif(self.optimizer in {"Mini_Batch_GD", "momentum","AdaGrad", "Adam", "RMSprop"}):
            for i in range(epoch):
                indices = np.random.permutation(n) # Randomly permute a sequence.
                x_shuffled = x[indices]
                z_hat_shuffled = z_hat[indices]
                for j in range(0, n, self.batch_size):
                    x_batch = x_shuffled[j:j+self.batch_size]
                    z_hat_batch = z_hat_shuffled[j:j+self.batch_size]
                    z = self.layer_output.forward(self.layer_2.forward(self.layer_1.forward(x_batch)))
                    loss, grad = compute_loss(z, z_hat_batch)
                    self.layer_1.backward(self.layer_2.backward(self.layer_output.backward(grad)))
                self.losses.append(loss)
                if(i % show_epoch == 0):
                    print(f"epoch {i} loss : {loss:.8f}")
        # GD, momentum, Adam, RMSprop
        elif(self.optimizer  == "GD"):
            for i in range(epoch):
                z = self.layer_output.forward(self.layer_2.forward(self.layer_1.forward(x)))
                loss, grad = compute_loss(z, z_hat)
                self.layer_1.backward(self.layer_2.backward(self.layer_output.backward(grad)))
                self.losses.append(loss)
                if(i % show_epoch == 0):
                    print(f"epoch {i} loss : {loss:.8f}")
        else:
            raise NameError("No such optimizer")

    def show_result(self, z_test, z_hat_test):
        for i in range(z_test.size):
            print(f"Iter{str(i + 1).rjust(3)} |    Ground truth: {z_hat_test[i]} |     prediction: {int(z_test[i][0] > 0.5)} |")
        loss, _ = compute_loss(z_test, z_hat_test)
        accuracy = np.sum((z_test > 0.5) == (z_hat_test==1)) / z_test.size
        print(f"loss={loss:.4f} Accuracy={accuracy:.2%}")
        display_result(self.losses, x_test, z_test > 0.5, z_hat_test, optimizer = self.optimizer)
        plt.show()



# Main function to run the neural network
if __name__ == "__main__":

    # Generate data, linear or XOR
    data_type = 'linear'
    if data_type == 'linear':
        X, z_hat = generate_linear(n=100)
        x_test, z_hat_test = generate_linear(n=100)
    else:
        X, z_hat = generate_XOR_easy()
        x_test, z_hat_test = generate_XOR_easy()

    # Train the network on the generated data #batch_size = 1(SGD)~Mini-Batch GD~100(GD)
    # activate: sigmoid, Relu, None (no activation)
    # optimizer: SGD, Mini_Batch_GD, GD; Mini_Batch_GD: momentum, AdaGrad, RMSprop, Adam
    # optimizer parameters: 
    # SGD, 
    # Mini_Batch_GD, 
    # GD; 
    # momentum, 
    # AdaGrad, 
    # RMSprop
    # Adam, 


    model = Model(neuron_of_hidden=5, lr=0.1, activate="sigmoid", optimizer="Mini_Batch_GD", batch_size = 10, bias=0) 
    model.train(X, z_hat, epoch=500, show_epoch=50)
    z_test = model.test(x_test)
    model.show_result(z_test, z_hat_test)
