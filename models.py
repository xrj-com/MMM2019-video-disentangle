import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv

# dcgan-64
def normal_init(m, mean, std):#D & G: mean=0, std=0.02
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

class ContentEncoder(nn.Module):
    # initializers
    def __init__(self, inD=3, d=64, contentDim=64, is_norm=True):
        super().__init__()
        self.is_norm = is_norm
        self.contentDim = contentDim
        self.conv1 = nn.Conv2d(inD, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d*4, d*8, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(d*8)
        self.conv5 = nn.Conv2d(d*8, contentDim, 4, 1, 0)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = F.tanh(self.conv5(x))
        if self.is_norm:
            x.view(-1, self.contentDim)
            x = F.normalize(x, p=2)
            x.view(-1, self.contentDim, 1, 1)
        return x

class PoseEncoder(nn.Module):
    # initializers
    def __init__(self, inD=3, d=64, poseDim=64, is_norm=True):
        super().__init__()
        self.is_norm = is_norm
        self.poseDim = poseDim
        self.conv1 = nn.Conv2d(inD, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d*4, d*8, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(d*8)
        self.conv5 = nn.Conv2d(d*8, poseDim, 4, 1, 0)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = F.tanh(self.conv5(x))
        if self.is_norm:
            x.view(-1, self.poseDim)
            x = F.normalize(x, p=2)
            x.view(-1, self.poseDim, 1, 1)
        return x

class Decoder(nn.Module):
    def __init__(self, inD=64+5, d=64):
        super().__init__()
        self.deconv1 = nn.ConvTranspose2d(inD, d*8, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(d*8)
        self.deconv2 = nn.ConvTranspose2d(d*8, d*4, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d*4)
        self.deconv3 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d*2)
        self.deconv4 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(d)
        self.deconv5 = nn.ConvTranspose2d(d, 3, 4, 2, 1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, sceneCode, poseCode):
        # x = F.relu(self.deconv1(input))
        input = torch.cat((sceneCode, poseCode), 1)
        x = F.relu(self.deconv1_bn(self.deconv1(input)))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.relu(self.deconv4_bn(self.deconv4(x)))
        x = F.sigmoid(self.deconv5(x))

        return x
        
class SceneDiscriminator(nn.Module):
    def __init__(self, poseDim=5, hidden_unit=100):
        super().__init__()
        self.poseDim = poseDim
        self.fc1 = nn.Linear(poseDim*2, hidden_unit)
        self.fc2 = nn.Linear(hidden_unit, hidden_unit)
        self.fc3 = nn.Linear(hidden_unit, 1)
    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, poseCode1, poseCode2):
        # x = F.relu(self.deconv1(input))
        x = torch.cat((poseCode1, poseCode2), 1)
        x = x.view(-1, self.poseDim*2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x

class lstm(nn.Module):
    def __init__(self, poseDim = 64, contentDim = 64, rnnSize, rnnLayers):
        self.inputDim = contentDim + poseDim
        self.hiddenSize = rnnSize
        self.lstm = nn.LSTM(self.inputDim, self.hiddenSize, rnnLayers)
        self.hidden = (torch.zeros(rnnLayers, 1, self.hiddenSize),
                        torch.zeros(rnnLayers, 1, self.hiddenSize))

    def forward(self, pose, content):
        x= torch.cat((content, pose), 1)
        poseCode, hidden = self.lstm(x, self.hidden)
        return poseCode

    #def __init__(self, poseDim = 64, contentDim = 64,rnnSize, rnnLayers):#FIXME:rnnLayers
    #    self.input_dim = poseDim + contentDim
    #    self.hidden_dim = rnnSize#FIXME
    #    self.lstm = nn.LSTM(self.input_dim, self.hidden_dim)#FIXME
    #    #self.fc = nn.Linear(poseDim+contentDim, rnnSize)
    #    #self.lstm = nn.LSTM(rnnSize, rnnSize)
    #    self.hidden = self.init_hidden()

    #def init_hidden(self):
    #    return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
    #            autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))

    #def forward(self, pose, content):
    #    x = torch.cat((content, pose), 1)
    #    #x = self.fc(x)#FIXME
    #    next_pose, self.hidden = self.lstm(x, self.hidden)
    #    return next_pose

        

if __name__ == "__main__":
    content = ContentEncoder()
    x = torch.randn(10, 3, 64, 64)
    print(x.size())
    contentcode = content(x)
    print(contentcode.size())

    post = PoseEncoder()
    postcode = content(x)
    print(postcode.size())

    D = Decoder(inD=64+64)
    ScenceD = SceneDiscriminator(poseDim=64) 
    out2 = ScenceD(postcode, postcode)
    out = D(contentcode, postcode)
    print(out.size())
    print(out2.size())

