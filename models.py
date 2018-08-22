import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
import torch.autograd as autograd
import copy

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


<<<<<<< HEAD
class lstmBlock(nn.Module):
    def __init__(self, poseDim = 5, contentDim = 64, rnnSize = 256, rnnLayers = 2,
            is_norm = True):
        super(lstmBlock, self).__init__()
        self.inputDim = contentDim + poseDim
        self.hiddenSize = rnnSize
        self.layers = rnnLayers
        self.lstm = nn.LSTM(self.inputDim, self.hiddenSize, self.layers)
        self.is_norm = is_norm
        self.fc = nn.Linear(self.hiddenSize, poseDim)

    def forward(self, pose, content, (h,c)):
        #pose = pose.reshape(1, 3, 5)
        x = torch.cat((content, pose), 2)#FIXME: need define in __init__ as a net?
        #print(tensor.size())
        #print(h.size())
        #print(c.size())
        #x = autograd.Variable(tensor)
        x, (hn, cn) = self.lstm(x, (h,c))
        poseCode = torch.tanh(self.fc(x))
        if self.is_norm == True:
            poseCode = F.normalize(poseCode, 2)

        return poseCode, (hn,cn)

class lstmNet(nn.Module):
    def __init__(self, poseDim = 5, contentDim = 64, rnnSize = 256, rnnLayers = 2,
            nPast = 10, nFuture = 10, T = 21, is_norm = True):
        super(lstmNet, self).__init__()
        self.lstms = [None]*T
        self.nPast = nPast
        self.nFuture = nFuture
        self.layers = rnnLayers
        self.hiddenSize = rnnSize

        for i in range(T):#FIXME:why need +1 block, used in bp
            self.lstms[i] = lstmBlock(poseDim, contentDim, rnnSize, rnnLayers, is_norm)

        
    #note: pose_reps is a seq, content is just an item
    def forward_obs(self, pose_reps, content, batchSize):#FIXME: need batchSize?
        #init hidden FIXME: hidden should be a member of lstm? and autograd.Variable?
        hidden = (autograd.Variable(torch.zeros(self.layers, batchSize, self.hiddenSize)),
                autograd.Variable(torch.zeros(self.layers, batchSize, self.hiddenSize)))
        gen_pose = []
        for i in range(self.nPast + self.nFuture):
            #x = torch.cat((content, pose_reps[i]), 1)
            tmp_pose, hidden = self.lstms[i](pose_reps[i], content, hidden)
            gen_pose.append(tmp_pose)
        return gen_pose, pose_reps
    
    def forward(self, pose, content, batchSize):
        hidden = (autograd.Variable(torch.zeros(self.layers, batchSize, self.hiddenSize)),
                autograd.Variable(torch.zeros(self.layers, batchSize, self.hiddenSize)))
        gen_pose = []
        in_pose = []
        for i in range(self.nPast + self.nFuture):
            if i < self.nPast:
                in_pose.append(pose[i])#FIXME: need deep copy?
            else:
                in_pose.append(gen_pose[i-1])

            tmp_pose, hidden = self.lstms[i](in_pose[i], content, hidden)
            #print('gen:  ', tmp_pose.size())
            gen_pose.append(tmp_pose)
        return gen_pose, in_pose
            

'''
class lstm(nn.Module):
    def __init__(self, poseDim = 5, contentDim = 64, rnnSize, rnnLayers = 2,
                nPast = 10, nFuture = 10, T, is_norm = True):
        self.inputDim = contentDim + poseDim
        self.hiddenSize = rnnSize
        self.layers = rnnLayers
        self.lstm = []
        self.hidden = []
        self.layer_out = []
        self.fc = nn.Linear(self.hiddenSize, poseDim)
        self.is_norm = is_norm
        for i in rnnLayers:
            self.lstm[i] = nn.LSTM(self.inputDim, self.hiddenSize, 1)
            #self.hiden[i] = self.init_hidden()
        self.init_hidden()
        #self.lstm = nn.LSTM(self.inputDim, self.hiddenSize, rnnLayers)
        #self.hidden = self.init_hidden()
    
    def init_hidden(self):
        #return (torch.zeros(1, 1, self.hiddenSize),#FIXME:not rnnLayers
        #        torch.zeros(1, 1, self.hiddenSize))
        for i in range(self.layers):
            self.hidden[i] = (autograd.Variable(torch.zeros(1, 1, self.hiddenSize)), 
                    autograd.Variable(torch.zeros(1, 1, self.hiddenSize)))

    def forward(self, pose, content):
        x= torch.cat((content, pose), 1)
        for i in range(self.layers):
            self.layer_out[i], hidden = self.lstm[i](x, self.hidden[i])
            x = torch.cat((content, self.layer_out[i]), 1)
        poseCode = F.tanh(self.fc(self.layer_out[self.layers-1]))
        if self.is_norm == True:
            poseCode = F.normalize(poseCode, 2)

        return poseCode
'''
=======
class RecDiscriminator(nn.Module):
    # initializers
    def __init__(self, inD=3, d=64):
        super().__init__()
        self.conv1 = nn.Conv2d(inD, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d*4, d*8, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(d*8)
        self.conv5 = nn.Conv2d(d*8, 1, 4, 1, 0)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input1, input2):

        input = torch.cat([input1, input2], 1)
        x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = F.sigmoid(self.conv5(x).view(-1, 1))
        return x
>>>>>>> master

if __name__ == "__main__":
    content = ContentEncoder()
    x = torch.randn(10, 3, 64, 64)
<<<<<<< HEAD
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

    lstm = lstmNet()
    contentCode = torch.randn(1, 3, 64)#(1,3,64)
    poseCode = [torch.randn(1, 3, 5)]*10# instead of (10,3,5) to avoid reshape
    gen_pose, in_pose = lstm(poseCode, contentCode, 3)
    print(len(gen_pose))
    print(len(in_pose))
    print(gen_pose[15].size())

=======
    print(torch.cat([x, x],1).size())
    # print(torch.cat([x,x],0).size())
    # print(x.size())
    # contentcode = content(x)
    # print(contentcode.size())

    # post = PoseEncoder()
    # postcode = content(x)
    # print(postcode.size())

    # RD = RecDiscriminator()
    # print(RD(x).size())


    # D = Decoder(inD=64+64)
    # ScenceD = SceneDiscriminator(poseDim=64) 
    # out2 = ScenceD(postcode, postcode)
    # out = D(contentcode, postcode)
    # print(out.size())
    # print(out2.size())
>>>>>>> master
