import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt

from ggnn import GGNN
from dataload import dataloader

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', type=str, default="processedData13C_NEW.pickle", help='path of input data')
parser.add_argument('--batchSize', type=int, default=128, help='input batch size')
parser.add_argument('--state_dim', type=int, default=28, help='GGNN hidden state size')
parser.add_argument('--n_node', type=int, default=44, help='Number of the input nodes')
parser.add_argument('--edge_types', type=int, default=10, help='types of the edge descriptor')
parser.add_argument('--n_steps', type=int, default=4, help='propogation steps number of GGNN')
parser.add_argument('--niter', type=int, default=65000, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--verbal', action='store_true', help='print training info or not')
#parser.add_argument('--manualSeed', type=int, help='manual seed')

opt = parser.parse_args()
print(opt)

# if opt.manualSeed is None:
#     opt.manualSeed = random.randint(1, 10000)
# print("Random Seed: ", opt.manualSeed)
# random.seed(opt.manualSeed)
# torch.manual_seed(opt.manualSeed)

#
#
# if opt.cuda:
#     torch.cuda.manual_seed_all(opt.manualSeed)


def train(epoch, dataset, net, criterion, optimizer, opt):
    net.train()
    start_ = (opt.batchSize * epoch) % len(dataset[0])
    size = opt.batchSize
    if (len(dataset[0]) - start_) < opt.batchSize:
        end_ = -1
        size = len(dataset[0]) - start_
    else:
        end_ = (start_ + opt.batchSize)
    #print(start_,end_)
    node_feat = torch.from_numpy(dataset[0][start_:end_, :, :])
    adj_matrix = torch.from_numpy(dataset[1][start_:end_, :, :])
    y = torch.from_numpy(dataset[2][start_:end_])
    annotation = torch.from_numpy(dataset[3][start_:end_])
    net.zero_grad()

    if opt.cuda:
        node_feat = node_feat.cuda()
        adj_matrix = adj_matrix.cuda()
        annotation = annotation.cuda()
        y = y.cuda()

    node_feat = Variable(node_feat).double()
    adj_matrix = Variable(adj_matrix).double()
    annotation = Variable(annotation).double()
    y = Variable(y).double()
    y_mask = torch.zeros_like(y)
    y_mask[torch.where(y>0)] = 1
    output = net(node_feat, annotation, adj_matrix, y_mask)

    loss = criterion(output, y)
    # regularization_loss = 0
    # for param in net.parameters():
    #      regularization_loss += torch.sum(abs(param))
    loss *= float(opt.n_node * size)
    # loss += regularization_loss
    loss /= float(torch.sum(y_mask))
    loss.backward()
    optimizer.step()

    if opt.verbal:
        #print(output)
        print('[%d/%d] Loss: %.4f' % (epoch, opt.niter, float(loss.data)))

def test(dataset, net, criterion, optimizer, opt):
    net.eval()
    torch.cuda.empty_cache()
    epoch = 0
    test_mae = 0
    y_masksum = 0
    while epoch * opt.batchSize < len(dataset[0]):
        start_ = (opt.batchSize * epoch)
        size = opt.batchSize
        if (len(dataset[0]) - start_) < opt.batchSize:
            end_ = -1
            size = len(dataset[0]) - start_
        else:
            end_ = (start_ + opt.batchSize)
        epoch += 1
        node_feat = torch.from_numpy(dataset[0][start_:end_, :, :])
        adj_matrix = torch.from_numpy(dataset[1][start_:end_, :, :])
        y = torch.from_numpy(dataset[2][start_:end_])
        annotation = torch.from_numpy(dataset[3][start_:end_])
        with torch.no_grad():
            if opt.cuda:
                node_feat = node_feat.cuda()
                adj_matrix = adj_matrix.cuda()
                annotation = annotation.cuda()
                y = y.cuda()

            node_feat = Variable(node_feat).double()
            adj_matrix = Variable(adj_matrix).double()
            annotation = Variable(annotation).double()
            y = Variable(y).double()
            y_mask = torch.zeros_like(y)
            y_mask[torch.where(y > 0)] = 1
            output = net(node_feat, annotation, adj_matrix, y_mask)

            mae = torch.sum(torch.abs(output - y))
            # loss = criterion(output, y)
            # loss *= float(opt.n_node * size)
            test_mae += mae
            y_masksum += torch.sum(y_mask)
    test_mae /= y_masksum

    print('Test_set Loss: %.4f' % (float(test_mae.data)))
    return(float(test_mae.data))


def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 2 epochs"""

    lr_ = lr * (0.33 ** (epoch // 6000))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_

def main(opt):
    train_dataset = dataloader("processedData13C_NEW.pickle","train",12582)
    # test_dataset = bAbIDataset(opt.dataroot, opt.question_id, False)
    # test_dataloader = bAbIDataloader(test_dataset, batch_size=opt.batchSize, \
    #                                  shuffle=False, num_workers=2)
    opt.annotation_dim = 1  # for bAbI
    #opt.n_edge_types = train_dataset.n_edge_types
    #opt.n_node = train_dataset.n_node

    net = GGNN(opt)
    net.double()
    print(net)


    criterion = nn.MSELoss()

    if opt.cuda:
        net.cuda()
        criterion.cuda()
    LR = opt.lr
    #optimizer = optim.Adam(net.parameters(), lr=LR )
    optimizer = optim.Adam(net.parameters(), lr=LR, weight_decay=0.0015)

    for epoch in range(0, opt.niter):
        train(epoch, train_dataset, net, criterion, optimizer, opt)
        #adjust_learning_rate(optimizer, epoch, LR)
        if epoch % 1000 == 0:
            test_dataset = dataloader("processedData13C_NEW.pickle", "val", 12582)
            test_mse = test(test_dataset, net, criterion, optimizer, opt)
            del(test_dataset)
            #adjust_learning_rate(optimizer, epoch, LR, test_mse)
            if test_mse < 1.95:
                print("TEST_SET")
                test_dataset1 = dataloader("processedData13C_NEW.pickle", "tst", 12582)
                test_mse1 = test(test_dataset1, net, criterion, optimizer, opt)
                del(test_dataset1)
            if test_mse < 1.92:
                torch.save(net, "d:/rotation2/datasets/ggnn_net3.pkl")
                print(epoch)
                #break
if __name__ == "__main__":
    main(opt)