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
import sklearn.metrics
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import tensorflow as tf
import pandas as pd
import datetime
def push_to_tensorboard(cm, f1_score, epoch, classes):
    currentTime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    val_log_dir = 'log_' + currentTime
    val_summary_writer = tf.summary.create_file_writer(val_log_dir)
    with val_summary_writer.as_default():
        tf.summary.scalar('f1_score', f1_score, step=epoch)
        tf.summary.image('confusion_matrix',
                         construct_confusion_matrix_image(classes, cm), step=epoch)


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

    buf = io.BytesIO()
    plt.savefig(buf, format='png')

    plt.close(figure)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)

    image = tf.expand_dims(image, 0)
    return image


def test(model, test_loader, btrain=False, model_file=None):
    if not btrain:
        model.load_state_dict(model_file)
    model.eval()

    correct = 0
    total = 0
    #
    class_correct = list(0. for i in range(6))
    class_total = list(0. for i in range(6))
    predictions = []
    ground_truth = []
    for i, (images, labels) in enumerate(test_loader):
        images = images.cuda()
        labels = labels.cuda()
        outputs = model(images)

        _, predicted = torch.max(outputs, 1)
        predicted_list = predicted.tolist()
        label_list = labels.tolist()
        predictions.extend(predicted_list)
        ground_truth.extend(label_list)

        total += labels.size(0)

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
    f1_score = sklearn.metrics.f1_score(np.array(ground_truth), np.array(predictions))
    confusion_matrix = sklearn.metrics.confusion_matrix(np.array(ground_truth), np.array(predictions))
    print('the f1 score is: '+str(f1_score))
    print('the confusion matrix is: ')
    print(confusion_matrix)
    return correct / total, f1_score, confusion_matrix

# train_dir = '/home/liushuai/cleaned_images/train'
# test_dir = '/home/liushuai/cleaned_images/validate'
# train_dir = '/data/sascha/Simcenter/cleaned_images/train'
# test_dir = '/data/sascha/Simcenter/cleaned_images/validate'
train_dir ='/home/liushuai/small_examples/images/train'
test_dir ='/home/liushuai/small_examples/images/validate'
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
lr = 0.1  # 0.1
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=0.0001)
is_train = True
acc_best = 0
total_epoch = 30 #TODO: change epoch
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
                e + 1, total_epoch, i + 1, len(train_loader), loss.data.item()))
        print('epoch {} takes time: '.format(e+1)+str(time.time() - tims))
        print('evaluate test set:')
        acc, f1, cm = test(model, test_loader, btrain=True)
        push_to_tensorboard(cm, f1, e, sorted(train_dataset.labels.keys()))
        if acc > acc_best:
            acc_best = acc
            print('current best acc,', acc_best)
            torch.save(model.state_dict(), model_file)
        if (e+1) / float(total_epoch) == 0.3 or (e+1) / float(total_epoch) == 0.6 or (e+1) / float(total_epoch) == 0.9:

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
    acc, f1, cm = test(model, test_loader, btrain=True, model_file=model_file)
    push_to_tensorboard(cm, f1, e, sorted(train_dataset.labels.keys()))