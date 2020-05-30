import torch
import torchvision
import torch.nn as nn
import os
from torchvision import transforms, models
from torch.utils.data import DataLoader
from residual_attention_network import ResidualAttentionModel_92
from preprocessing import GoogleStreetView
import torch.optim as optim
from torch.autograd import Variable
import time
def test(model, test_loader, btrain=False, model_file=None):
    if not btrain:
        model.load_state_dict(model_file)
    model.eval()

    correct = 0
    total = 0
    #
    class_correct = list(0. for i in range(6))
    class_total = list(0. for i in range(6))

    for images, labels in test_loader:
        images = Variable(images.cuda())
        labels = Variable(labels.cuda())
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.data).sum()
        #
        c = (predicted == labels.data).squeeze()
        for i in range(20):
            label = labels.data[i]
            class_correct[label] += c[i]
            class_total[label] += 1

    print('Accuracy of the model on the test images: %d %%' % (100 * float(correct) / total))
    print('Accuracy of the model on the test images:', float(correct) / total)
    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            i, 100 * class_correct[i] / class_total[i]))
    return correct / total


transform = transforms.Compose([transforms.Resize((224,224)), transforms.RandomHorizontalFlip])
train_dataset = GoogleStreetView(os.path.join('/home/liushuai/cleaned_images/train', 'description_train.csv'), transform=transform)
test_dataset = GoogleStreetView(os.path.join('/home/liushuai/cleaned_images/validate', 'description_test.csv')
                                , transform=transform, labels=train_dataset.labels)
train_loader = DataLoader(train_dataset,  batch_size=4,
                        shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset,  batch_size=20,
                        shuffle=True, num_workers=4)
model = ResidualAttentionModel_92().cuda()
print(model)
model_file = 'best_accuracy.pkl'
is_pretrain = False
lr = 0.1  # 0.1
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=0.0001)
is_train = True
acc_best = 0
total_epoch = 1 #TODO: change epoch
if is_train is True:
    if is_pretrain == True:
        model.load_state_dict((torch.load(model_file)))
    for e in range(total_epoch):
        model.train()
        tims = time.time()
        for i, (images, labels) in enumerate(train_loader):
            images = Variable(images.cuda())
            # print(images.data)
            labels = Variable(labels.cuda())

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print("hello")
            if (i + 1) % 100 == 0:
                print("Epoch [%d/%d], Iter [%d/%d] Loss: %.4f" % (
                e + 1, total_epoch, i + 1, len(train_loader), loss.data[0]))
        print('epoch {} takes time: '.format(e+1)+str(time.time() - tims))
        print('evaluate test set:')
        acc = test(test, test_loader, btrain=True)
        if acc > acc_best:
            acc_best = acc
            print('current best acc,', acc_best)
            torch.save(model.state_dict(), model_file)
        if e in [2,5,8]:
            lr /= 10
            print('reset learning rate to:', lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
                print(param_group['lr'])
            # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            # optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=0.0001)
            # Save the Model
    torch.save(model.state_dict(), 'last_model_92_sgd.pkl')
else:
    test(model, test_loader, btrain=False, model_file=model_file)