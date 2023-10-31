import torch
import torch.nn as nn

class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        
        self.padding = kernel_size // 2
        
        self.Wxi = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whi = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxf = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whf = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxg = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whg = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxo = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Who = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        
    def forward(self, x, h_prev, c_prev):
        i = torch.sigmoid(self.Wxi(x) + self.Whi(h_prev))
        f = torch.sigmoid(self.Wxf(x) + self.Whf(h_prev))
        g = torch.tanh(self.Wxg(x) + self.Whg(h_prev))
        o = torch.sigmoid(self.Wxo(x) + self.Who(h_prev))
        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next

class ImagePredictor(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, sequence_length):
        super(ImagePredictor, self).__init__()
        
        self.sequence_length = sequence_length
        self.hidden_channels = hidden_channels
        
        self.convlstm = ConvLSTMCell(input_channels, hidden_channels, kernel_size)
        self.conv = nn.Conv2d(hidden_channels, input_channels, kernel_size=1)  # To get the next frame prediction
        
    def forward(self, x):
        b, seq_len, c, h, w = x.size()
        
        h_t, c_t = torch.zeros(b, self.hidden_channels, h, w).to(x.device), torch.zeros(b, self.hidden_channels, h, w).to(x.device)
        for t in range(self.sequence_length):
            h_t, c_t = self.convlstm(x[:, t], h_t, c_t)
            
        # Predict the next frame
        return self.conv(h_t)

import numpy as np

def generate_data(batch_size, sequence_length, image_size, max_step_size=2):
    # Create a random direction and starting point for each sequence in the batch
    directions = np.random.randint(0, 4, batch_size)  # 0: right, 1: down, 2: left, 3: up
    start_points = np.random.randint(0, image_size, (batch_size, 2))
    
    sequences = np.zeros((batch_size, sequence_length, 1, image_size, image_size))
    
    for b in range(batch_size):
        x, y = start_points[b]
        dx, dy = 0, 0
        
        if directions[b] == 0: dx = max_step_size
        if directions[b] == 1: dy = max_step_size
        if directions[b] == 2: dx = -max_step_size
        if directions[b] == 3: dy = -max_step_size
        
        for t in range(sequence_length):
            sequences[b, t, 0, y, x] = 1.0
            x = np.clip(x + dx, 0, image_size - 1)
            y = np.clip(y + dy, 0, image_size - 1)
    
    return torch.tensor(sequences, dtype=torch.float32)

import torch.optim as optim

if __name__ == "__main__":
    # Hyperparameters
    batch_size = 64
    sequence_length = 32
    image_size = 32
    input_channels = 1  # Grayscale images
    hidden_channels = 64
    kernel_size = 3
    lr = 0.001
    num_epochs = 100
    print_interval = 10

    # Initialize model, loss, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImagePredictor(input_channels, hidden_channels, kernel_size, sequence_length).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        
        # Generate training data
        sequences = generate_data(batch_size, sequence_length+1, image_size).to(device)  # +1 for the target frame
        inputs = sequences[:, :sequence_length]  # First 32 frames
        targets = sequences[:, sequence_length]  # The 33rd frame is our target

        # Forward pass
        outputs = model(inputs)
        
        # Loss and optimization
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print updates
        if (epoch + 1) % print_interval == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
    
    # Save the model's state_dict to a file
    torch.save(model.state_dict(), 'image_predictor.pth')
    
    print("Training finished!")
