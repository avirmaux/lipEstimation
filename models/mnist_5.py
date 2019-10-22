import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms

from torch.autograd import Variable

from utils import load_model

def mnist_5(loc=None):
    """
    So far, LB for Lipschitz constant on this model is 25.5323 obtained by
    annealing
    """
    mnist = MNIST_classifier()
    if loc is None:
        loc = 'models/mnist_5.pth.tar'
    load_model(mnist, loc)
    return mnist

def test(model, dataset, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in dataset:
        # data, target = data, target
        data, target = Variable(data).cuda(), Variable(target).cuda()
        output = model(data)
        test_loss += loss_fn(output, target)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).sum()

        test_loss /= len(dataset)
    print("Test set: Average loss: {:.3f},\
            Accuracy: {}/{} ({:.2f}%)\n".format(
                test_loss,
                correct,
                len(dataset.dataset),
                100. * float(correct) / float(len(dataset.dataset))))

class MNIST_classifier(nn.Module):

    def __init__(self):
        super(MNIST_classifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, stride=2)

        self.conv5 = nn.Conv2d(128, 10, 2)

        self.act = nn.ReLU(inplace=True)
        # self.act = nn.Sigmoid()

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = self.act(self.conv4(x))
        x = self.conv5(x).view(-1, 10)
        return x

if __name__ == '__main__':

    use_cuda = torch.cuda.is_available()

    batch_size = 256
    loss_fn = nn.CrossEntropyLoss()
    clf = MNIST_classifier()
    if use_cuda:
        clf = clf.cuda()
    optimizer = optim.Adam(clf.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2)

    data_train = datasets.MNIST(root='data/', download=True, train=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

    train_load = torch.utils.data.DataLoader(data_train, batch_size=batch_size,
            shuffle=True, num_workers=4)

    data_test = datasets.MNIST(root='data/', download=True, train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

    train_test = torch.utils.data.DataLoader(data_test, batch_size=batch_size,
            shuffle=True, num_workers=4)

    epoch = 1

    while epoch < 40:
        for (idx, (data, target)) in enumerate(train_load):
            data, target = Variable(data), Variable(target)
            if use_cuda:
                data, target = data.cuda(), target.cuda()

            optimizer.zero_grad()

            out = clf(data)
            loss = loss_fn(out, target)
            loss.backward()
            optimizer.step()
            if idx % 10 == 0:
                print('[Epoch {} | {}/{}]: {:.4f}'.format(epoch,
                    idx, len(train_load),
                    loss))
        epoch += 1
        scheduler.step()
        test(clf, train_test, epoch)
