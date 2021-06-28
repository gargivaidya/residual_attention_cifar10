### YOUR CODE HERE
# import tensorflow as tf
import torch
import os, time
import numpy as np
#from Network import MyNetwork
from ImageUtils import parse_record

#from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchvision
from torchvision import transforms, datasets, models
import os
import cv2
import time
import gc
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
writer = SummaryWriter('logs/')

from Network import ResidualAttentionNetwork

"""This script defines the training, validation and testing process.
"""

class MyModel(object):

    def __init__(self, configs):
        self.configs = configs
        self.network = ResidualAttentionNetwork().cuda()

    def model_setup(self):
        print('---Setup input interfaces...')

        print('---Setup the network...')
        #model = ResidualAttentionNetwork().cuda()
        model = self.network
        print(model)

        if training:
            print('---Setup training components...')

            print('---Setup the Saver for saving models...')
            self.saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=0)
        else:
            print('---Setup testing components...')

            print('---Setup the Saver for loading models...')
            self.loader = tf.train.Saver(var_list=tf.global_variables())
        

    def train(self, x_train, y_train, configs, x_valid=None, y_valid=None):
        model = ResidualAttentionNetwork().cuda()
        #print(model)
        #writer.add_graph(model, x_train)
        
        classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        lr = 0.1
        batch_size = 64
        max_epoch = 5
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=0.0001)
        print('###Train###')
        num_samples = x_train.shape[0]
        num_batches = int(num_samples / batch_size)

        print('---Run...')

        for epoch in tqdm(range(1, max_epoch+1)):
            model.train()
            start_time = time.time()

            # Shuffle
            shuffle_index = np.random.permutation(num_samples)
            curr_x_train = x_train[shuffle_index]
            curr_y_train = y_train[shuffle_index]

            for i in range(num_batches):
                
                # Construct the current batch.
                # Don't forget to use "parse_record" to perform data preprocessing.
                x_batch = []
                for batch_i in range(batch_size*i, batch_size*(i+1)):
                    x_batch.append(parse_record(curr_x_train[batch_i], True))
                x_batch = np.array(x_batch).reshape((batch_size, 32, 32, 3))
                y_batch = curr_y_train[batch_size*i:batch_size*(i+1)]

                x_batch = torch.from_numpy(x_batch).cuda()
                y_batch = torch.from_numpy(y_batch).cuda()
                y_batch = torch.tensor(y_batch, dtype=torch.long)
                x_batch = x_batch.permute(0,3,1,2)

                #print(x_batch.shape)

                optimizer.zero_grad()
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                print('Batch {:d}/{:d} Loss {:.6f}'.format(i, num_batches, loss),
                        end='\r', flush=True)

                gc.collect()
                torch.cuda.empty_cache()               

            acc = self.evaluate(x_valid, y_valid)
            if acc > acc_best:
                acc_best = acc
                print('current best acc,', acc_best)
                torch.save(model.state_dict(), model_file)   

            writer.add_scalar('loss/train', np.random.random(), i)
            writer.add_scalar('acc/train', np.random.random(), i)

            if (epoch+1) / float(total_epoch) == 0.3 or (epoch+1) / float(total_epoch) == 0.6 or (epoch+1) / float(total_epoch) == 0.9:
                lr /= 10
                print('reset learning rate to:', lr)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                    print(param_group['lr'])            

            duration = time.time() - start_time
            print('Epoch {:d} Loss {:.6f} Duration {:.3f} seconds.'.format(
                        epoch, loss, duration))

    def evaluate(self, x, y):
        gc.collect()

        model = self.network
        print("here")
        
        correct = 0
        total = 0
        
        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))

        x_test=[]

        for i in range(x.shape[0]):
            x_test.append(parse_record(x[i], False))
            
        labels = y
        x_test = np.array(x_test).reshape((x.shape[0], 32, 32, 3))

        x_test = torch.from_numpy(x_test).cuda()
        x_test = x_test.permute(0,3,1,2)

        #print(x_test.shape)
        #print(labels.shape)
        gc.collect()
        torch.cuda.empty_cache()

        outputs = model(x_test)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.data).sum()
        
        c = (predicted == labels.data).squeeze()
        for i in range(20):
            label = labels.data[i]
            class_correct[label] += c[i]
            class_total[label] += 1

        print('Accuracy of the model on the test images: %d %%' % (100 * float(correct) / total))
        print('Accuracy of the model on the test images:', float(correct)/total)
        for i in range(10):
            print('Accuracy of %5s : %2d %%' % (
                classes[i], 100 * class_correct[i] / class_total[i]))
        return correct / total

    def predict_prob(self, x):
        model = ResidualAttentionNetwork()
        model_file = 'res_attn_model.pkl'
        model.load_state_dict(torch.load(model_file))
        model.eval()
        test_loader_1 = torch.utils.data.DataLoader(dataset=x,
                                          batch_size=20,
                                          shuffle=False)
        pred_proba = Variable()
        for images in test_loader_1:
            images = Variable(images)
            #print(images.shape)
            images = images.float()
            #model = model.cpu()
    
            outputs = model(images)
            #print(outputs.shape)
            outputs = outputs.cpu()
            pred_proba = torch.cat((pred_proba,outputs),0)
        print(pred_proba.shape)
        with open('predictions.npy', 'wb') as f:
            pred_proba = pred_proba.detach().numpy()
            np.save(f, pred_proba)
        return pred_proba   

### END CODE HERE