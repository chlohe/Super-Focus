from torch import nn
from torch.nn.modules.activation import LeakyReLU
from torchvision.models.resnet import resnet50


class Discriminator(nn.Module):

    def __init__(self):
        super().__init__()
        self.net = resnet50()
        self.net.conv1 = nn.Conv2d(1, 64, kernel_size=(
            7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.net.fc = nn.Linear(in_features=2048, out_features=2, bias=True)
        self.softmax = nn.Softmax()

        def make_relu_leaky(model):
            for child_name, child in model.named_children():
                if isinstance(child, nn.ReLU):
                    setattr(model, child_name, nn.LeakyReLU())
                else:
                    make_relu_leaky(child)
        make_relu_leaky(self.net)

    def forward(self, x):
        x = self.net(x)
        x = self.softmax(x)
        return x
