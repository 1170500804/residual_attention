import torch
import torchvision
import torch.nn as nn
import os

from sklearn.metrics import classification_report, precision_recall_fscore_support
from torchvision import transforms, models
from torch.utils.data import DataLoader
from residual_attention_network import ResidualAttentionModel_92
from preprocessing import GoogleStreetView
import torch.optim as optim
from torch.autograd import Variable
import time
import sklearn.metrics
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import datetime
import argparse

parser = argparse.ArgumentParser(description='Train Residual Attention')
parser.add_argument('--train-data', default='/home/liushuai/small_examples/images/train', type=str,
                    help='Folder containing train data')
parser.add_argument('--val-data', default='/home/liushuai/small_examples/images/validate', type=str,
                    help='Folder containing validation data')
args = parser.parse_args()

train_dir = args.train_data
test_dir = args.val_data

currentTime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
val_log_dir = os.path.join('runs','log_' + currentTime)
summary_writer = SummaryWriter(val_log_dir)

def construct_confusion_matrix_image(classes, con_mat):
    con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)

    con_mat_df = pd.DataFrame(con_mat_norm,
                              index=classes,
                              columns=classes)

    figure = plt.figure(figsize=(8, 8))
    sns.heatmap(con_mat_df, annot=True, cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    figure.canvas.draw()

    # Now we can save it to a numpy array.
    data_confusion_matrix = np.fromstring(figure.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data_confusion_matrix = data_confusion_matrix.reshape(figure.canvas.get_width_height()[::-1] + (3,))

    data_confusion_matrix = torch.from_numpy(np.transpose(data_confusion_matrix, (2, 0, 1)))

    return data_confusion_matrix


def test(model, criterion, test_loader, btrain=False, model_file=None):
    if not btrain:
        model.load_state_dict(model_file)
    model.eval()

    correct = 0
    total = 0
    #
    class_correct = list(0. for i in range(6))
    class_total = list(0. for i in range(6))
    predictions = []
    ground_truths = []
    avg_val_loss = 0
    for i, (images, labels) in enumerate(test_loader):
        images = images.cuda()
        labels = labels.cuda()
        outputs = model(images)

        _, predicted = torch.max(outputs, 1)
        predicted_list = predicted.tolist()
        label_list = labels.tolist()
        predictions.extend(predicted_list)
        ground_truths.extend(label_list)

        total += labels.size(0)
        loss = criterion(outputs, labels)

        avg_val_loss += loss.detach().cpu().numpy()

        correct += (predicted == labels.data).sum()

        c = (predicted == labels).squeeze()
        for i in range(labels.size(0)):
            label = labels[i]
            if labels.size(0) == 1:
                class_correct[label] += c.item()
            else:
                class_correct[label] += c[i]
            class_total[label] += 1


    print('Accuracy of the model on the test images: %d %%' % (100 * float(correct) / total))
    print('Accuracy of the model on the test images:', float(correct) / total)
    for i in range(6):
        print('Accuracy of %5s : %2d %%' % (
            i+5001, 100 * class_correct[i] / class_total[i]))
    f1_score = sklearn.metrics.f1_score(np.array(ground_truths), np.array(predictions),average='micro')
    confusion_matrix = sklearn.metrics.confusion_matrix(np.array(ground_truths), np.array(predictions))

    summary_writer.add_scalar('Val/Loss', avg_val_loss.mean(), epoch)
    report_items = precision_recall_fscore_support(ground_truths, predictions, average='micro')
    summary_writer.add_scalar('Val/F1', report_items[2], epoch)
    summary_writer.add_scalar('Val/Precision', report_items[0], epoch)
    summary_writer.add_scalar('Val/Recall', report_items[1], epoch)

    print('the f1 score is: '+str(f1_score))
    print('the confusion matrix is: ')
    print(confusion_matrix)
    return correct / total, f1_score, confusion_matrix


transform = transforms.Compose([transforms.Resize((224,224)), transforms.RandomHorizontalFlip(), transforms.ToTensor()])
train_dataset = GoogleStreetView(os.path.join(train_dir, 'description_train.csv'), transform=transform)
test_dataset = GoogleStreetView(os.path.join(test_dir, 'description_test.csv')
                                , transform=transform, labels=train_dataset.labels)
train_loader = DataLoader(train_dataset,  batch_size=8,
                        shuffle=True, num_workers=8)
test_loader = DataLoader(test_dataset,  batch_size=8,
                        shuffle=True, num_workers=4)
model = ResidualAttentionModel_92().cuda()
print(model)
model_file = 'best_accuracy.pkl'
is_pretrain = False
lr = 0.001  # 0.1
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=0.0001)
#optimizer = optim.Adam(model.parameters())#, lr=lr, momentum=0.9, nesterov=True, weight_decay=0.0001)
is_train = True
best_f1 = 0
total_epochs = 30 #TODO: change epoch
if is_train is True:
    if is_pretrain == True:
        model.load_state_dict((torch.load(model_file)))
    for epoch in range(total_epochs):
        model.train()
        tims = time.time()
        avg_train_loss = 0

        predictions =  []
        ground_truths = []

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

            avg_train_loss += loss.detach().cpu().numpy()
            _, predicted = torch.max(outputs, 1)

            predictions.extend(predicted.tolist())
            ground_truths.extend(labels.tolist())

            if (i + 1) % 100 == 0:
                print("Epoch [%d/%d], Iter [%d/%d] Loss: %.4f" % (
                    epoch + 1, total_epochs, i + 1, len(train_loader), loss.data.item()))

        confusion_matrix = sklearn.metrics.confusion_matrix(np.array(ground_truths), np.array(predictions))
        summary_writer.add_image('Train/confusion_matrix',
                                 construct_confusion_matrix_image(train_dataset.classes, confusion_matrix), epoch)
        print('epoch {} takes time: '.format(epoch + 1) + str(time.time() - tims))
        summary_writer.add_scalar('Train/Loss', avg_train_loss.mean(), epoch)
        report_items = precision_recall_fscore_support(ground_truths, predictions,average='micro')
        summary_writer.add_scalar('Train/F1', report_items[2], epoch)
        summary_writer.add_scalar('Train/Precision', report_items[0], epoch)
        summary_writer.add_scalar('Train/Recall', report_items[1], epoch)

        print('evaluate test set:')
        acc, f1, cm = test(model, criterion, test_loader, btrain=True)
        summary_writer.add_image('Val/confusion_matrix',
                                 construct_confusion_matrix_image(train_dataset.classes, cm), epoch)

        if f1 > best_f1:
            best_f1 = f1
            print('current best f1,', best_f1)
            torch.save(model.state_dict(), model_file)
        if (epoch + 1) / float(total_epochs) == 0.3 or (epoch + 1) / float(total_epochs) == 0.6 or (epoch + 1) / float(total_epochs) == 0.9:

            lr /= 10
            print('reset learning rate to:', lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
                print(param_group['lr'])
            # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            # optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=0.0001)
            # Save the Model
        torch.save(model.state_dict(), 'checkpoint.pkl')
else:
    acc, f1, cm = test(model, test_loader, btrain=True, model_file=model_file)
    raise NotImplementedError('Sorry I made this part unusable for now')
    # Sorry I made this part unusable for now
    #push_to_tensorboard(cm, f1, 1, sorted(train_dataset.labels.keys()))