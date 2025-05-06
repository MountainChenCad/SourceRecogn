import torch
import torch.nn as nn
import torch.nn.functional as F

# --- ResNet 1D ---
# Based on standard ResNet, adapted for 1D convolutions

class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=7, stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet1D(nn.Module):
    def __init__(self, block, num_blocks, num_classes=8, input_channels=2):
        super(ResNet1D, self).__init__()
        self.in_planes = 64

        # Initial convolution adapted for potentially longer sequences
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=15, stride=2, padding=7, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1) # Global Average Pooling
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # Input x shape: (batch, channels=2, length)
        out = F.relu(self.bn1(self.conv1(x)))
        # Consider adding MaxPool1d here if needed
        # out = F.max_pool1d(out, kernel_size=3, stride=2, padding=1)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        features = torch.flatten(out, 1) # Save features before classification head
        out = self.fc(features)
        return out

    def get_features(self, x):
         # Helper to get features before the final layer (useful for some transfer learning analysis)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        features = torch.flatten(out, 1)
        return features

def ResNet18_1D(num_classes=8):
    return ResNet1D(BasicBlock1D, [2, 2, 2, 2], num_classes=num_classes)

# --- LSTM Model ---

class LSTMNet(nn.Module):
    def __init__(self, input_size=2, hidden_size=128, num_layers=2, num_classes=8, dropout=0.2):
        super(LSTMNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # batch_first=True means input/output shape is (batch, seq_len, feature)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0, bidirectional=False)
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, num_classes) # Use hidden_size * 2 if bidirectional=True

    def forward(self, x):
        # Input x shape: (batch, seq_len, features=2)
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        out, _ = self.lstm(x, (h0, c0))

        # Decode the hidden state of the last time step
        # out shape: (batch_size, hidden_size)
        last_hidden_state = out[:, -1, :]

        # Pass through linear layer
        out = self.fc(last_hidden_state)
        return out

    def get_features(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        last_hidden_state = out[:, -1, :]
        return last_hidden_state


# --- Model Getter ---
def get_model(model_type, num_classes):
    if model_type == 'resnet':
        print("Using ResNet18-1D model.")
        return ResNet18_1D(num_classes=num_classes)
    elif model_type == 'lstm':
        print("Using LSTM model.")
        # You might want to expose hidden_size, num_layers as args
        return LSTMNet(num_classes=num_classes, hidden_size=128, num_layers=2)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
