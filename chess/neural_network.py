import torch
import torch.nn as nn

class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        # 在 ChessNet 类的 __init__ 方法中
        self.conv1 = nn.Conv2d(in_channels=14, out_channels=256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(256)

        self.policy_conv = nn.Conv2d(256, 2, 1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 10 * 9, 90 * 90)

        self.value_conv = nn.Conv2d(256, 1, 1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(10 * 9, 256)
        self.value_fc2 = nn.Linear(256, 1)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))

        policy = self.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(-1, 2 * 10 * 9)
        policy = self.policy_fc(policy)

        value = self.relu(self.value_bn(self.value_conv(x)))
        value = value.view(-1, 10 * 9)
        value = self.relu(self.value_fc1(value))
        value = self.tanh(self.value_fc2(value))

        return policy, value