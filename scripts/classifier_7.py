
##################################
# example with 4 categories and test and validation split

# globals().clear()
import sys
import torch.utils.data as data_utils
from torch import nn
import torch
import time
import numpy as np
from nltk.stem.snowball import SnowballStemmer

sys.path.append('D:/info/uni_tubingen/02 Master/4. Semester/Thesis/Thesis/scripts/data_preprocessing/training_datasets')
sys.path.append('D:/info/uni_tubingen/02 Master/4. Semester/Thesis/Thesis/scripts/')

from global_custom_functions import clean_and_split_text



from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import os
import datetime

###################################
# load training data

import pandas as pd
import matplotlib.pyplot as plt
os.getcwd()



class DictionaryData(Dataset):
    def __init__(self, data):
        """
        Args:
        """
        self.data = None
        self.labels = None
        self.transform_dataset(data)
        self.get_labels(data)

    def transform_dataset(self, data):
        self.data = data.sentence.to_list()

    def get_labels(self, data):
        self.labels = data.label.to_list()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = list(self.data)[idx]
        return sample

    def __getlabel__(self, idx):
        return self.labels[idx]


class TextClassificationModel(nn.Module):

    def __init__(self, vocab_size, embed_dim, num_class, batch_size=21):
        super(TextClassificationModel, self).__init__()
        # self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(5756, 3000)
        self.tanh = nn.Tanh()
        self.fc_2 = nn.Linear(3000, num_class)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        # self.fc_3 = nn.Linear(500, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        # self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
        # self.fc_2.weight.data.uniform_(-initrange, initrange)
        self.fc_2.bias.data.zero_()
        # self.conv.weight.data.uniform_(-initrange, initrange)
        # self.fc_3.weight.data.uniform_(-initrange, initrange)
        torch.nn.init.xavier_uniform_(self.fc_2.weight)
        torch.nn.init.xavier_uniform_(self.fc.weight)
        # torch.nn.init.xavier_uniform_(self.fc_3.weight)
        # self.fc_3.bias.data.zero_()

    def forward(self, text):
        text.shape
        text = text.type(torch.float)
        t = text.unsqueeze(0)
        t.shape
        # return self.fc_2(self.tanh_2(self.conv(self.tanh(self.fc(t)))))
        return (self.fc_2(self.relu(self.tanh(self.fc(t)))))


class Model():
    ############3
    # 1. define the data

    def __init__(self):
        self.dataloader_train = None
        self.dataloader_testt = None
        self.dataloader_valid = None
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.vectorizer = None
        self.batch_size = None


    def load_datasets(self):
        training_data = pd.DataFrame(columns=['sentence', 'jobTitle', 'company', 'estimatedDatePosted', 'label'])

        for file in os.listdir('./data_preprocessing/training_datasets/'):
            # print(file)
            data = pd.read_csv(f'./data_preprocessing/training_datasets/{file}').drop('Unnamed: 0', axis=1)
            training_data = pd.concat([data, training_data])
            # print(training_data.shape)
            training_data = training_data.reset_index().drop(['index'],axis=1)
        print('Nr. of loaded sentences for training: ', training_data.shape[0])

        data_labels = training_data.label.unique()
        target_labels = {}
        for i, name in enumerate(data_labels):
            target_labels[name] = i


        #########################
        # 2. Create Dataset object
        dic_data = DictionaryData(training_data)

        # dic_data.__len__()
        dic_data.__getitem__(800)
        dic_data.__getlabel__(800)




        #########################
        # 2. Create DataLoader object

        #########################
        # 3. Show what itterating thru the dataloader gives:


        counter = Counter()
        for (line) in list(dic_data.data):
            print(line)
            counter.update(line.split())
        vocab = Vocab(counter, min_freq=2)



        features = list(dic_data.data)


        b = 'this is air'
        def text_pipeline(x, vocab):
            pipe = []
            # eng_stem = SnowballStemmer('english')
            ger_stem = SnowballStemmer('german')
            # for t in tokenizer(x):
                # if len(ger_stem.stem(t)) < len(eng_stem.stem(t)):
                #     word = ger_stem.stem(t)
                # else:
                #     word = eng_stem.stem(t)


                # word = t
                # print(word)
                # pipe.append(vocab[word])

            for t in x.split():
                word = ger_stem.stem(t)
                pipe.append(vocab[word])

            return pipe
        text_pipeline(b, vocab)

        features = []
        for i in list(dic_data):
            features.append(text_pipeline(i,vocab))
        ##########################################
        # Feature representation:
        # Each sentence must be a vector from Tf-Idf. Each sentence must have the same length
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(stop_words='english')
        tf_idf_matrix = vectorizer.fit_transform(dic_data.data)
        print(tf_idf_matrix.shape)
        vectorizer.vocabulary_
        vectorizer_features = vectorizer.get_feature_names()
        len(vectorizer_features)

        df = pd.DataFrame(tf_idf_matrix.toarray(), columns = vectorizer.get_feature_names())
        print(df)

        a = tf_idf_matrix[1,:]
        print(a)

        mat = tf_idf_matrix.toarray()
        mat[5,:].nonzero()


        dic_data.data[1200]
        dic_data.__getlabel__(1200)


        target = []
        nr_skills = 0

        for i, text in enumerate(list(dic_data.data)):
            print(i)
            label = dic_data.__getlabel__(i)
            target.append(int(target_labels[label]))
            if label == 'skillset':
                nr_skills +=1

        print("Percentage of 'skills' in job ads: ", nr_skills/len(list(dic_data.data)))



        ##############################3
        # transform to tensors
        # features = torch.tensor(features)
        features = torch.tensor(mat)
        target = torch.tensor(target)

        print('Shape feat:', features.shape[1])
        print('Shape target (nr. sentences):', target.shape)



        def find_optimal_batch_size(features_size):
            for i in range(1,features_size-1):
                if (features_size) % i  == 0:
                    if i>100 and i <1000:
                        # print(f"size of batch: {i}, nr. of batches: {(features_size) / i}")
                        if (features_size)/i>=8:
                            print(f"size of batch: {i}, nr. of batches: {(features_size)/i}")
                            return i, (features_size)/i


        features_size_train = int(np.round(len(features)))
        print(features_size_train)
        BATCH_SIZE, total_batches = find_optimal_batch_size(features_size_train)
        self.batch_size = BATCH_SIZE

        split = int(BATCH_SIZE*np.floor(total_batches*0.7))
        indices = list(range(mat.shape[0]))
        train_split, test_split =  indices[:split], indices[split:]


        train = data_utils.TensorDataset(features, target)
        train_data = data_utils.TensorDataset(features[train_split], target[train_split])
        test_data = data_utils.TensorDataset(features[test_split], target[test_split])
        dataloader_train = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
        dataloader_test = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

        for i_batch, (sample_batched,label) in enumerate(dataloader_train):
            print(i_batch, sample_batched.shape)

        for i_batch, (sample_batched,label) in enumerate(dataloader_test):
            print(i_batch, sample_batched.shape)


        self.dataloader_train = dataloader_train
        self.dataloader_test = dataloader_test
        self.vectorizer = vectorizer

    def train(self, epoch):
        self.model.train()
        total_acc, total_count = 0, 0
        log_interval = 10
        start_time = time.time()
        # optimizer.zero_grad()
        loss_training = []

        for idx, (text, label) in enumerate(self.dataloader_train):
            self.optimizer.zero_grad()

            text = text.type(torch.float)
            label = label.type(torch.long)

            predited_label = self.model(text)
            predited_label = predited_label[0]
            # predited_label = predited_label.reshape(1,3)[0]

            loss = self.criterion(predited_label, label)
            loss_training.append(loss.detach().numpy())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
            self.optimizer.step()
            total_acc += (predited_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
            if idx % log_interval == 0 and idx > 0:
                print('Predicted labels:', [predited_label[:2]])
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches '
                      '| accuracy {:8.3f}'.format(epoch, idx, len(self.dataloader_train),
                                                  total_acc/total_count))
                total_acc, total_count = 0, 0
                start_time = time.time()
        return loss_training

    def evaluate(self):
        self.model.eval()
        total_acc, total_count = 0, 0
        loss_eval = []

        with torch.no_grad():
            for idx, (text, label) in enumerate(self.dataloader_test):
                text = text.type(torch.float)
                label = label.type(torch.long)
                # label = label.reshape(5,1)
                predited_label = self.model(text)
                predited_label = predited_label[0]

                loss = self.criterion(predited_label, label)
                total_acc += (predited_label.argmax(1) == label).sum().item()
                total_count += label.size(0)
                loss_eval.append(loss.detach().numpy())
        return total_acc/total_count, loss_eval

    def accuracy_per_category(self):
        total_acc, total_count = 0, 0
        dic_correct = {1: 0, 2: 0, 3: 0, 4: 0}
        dic_total = {1: 0, 2: 0, 3: 0, 4: 0}
        dic_accuracy = {1: 0, 2: 0, 3: 0, 4: 0}

        for idx, (text, label) in enumerate(self.dataloader_test):
            text = text.type(torch.float)
            label = label.type(torch.long)
            # label = label.reshape(5,1)
            predited_label = self.model(text)
            predited_label = predited_label[0]

            for i in range(predited_label.shape[0]):
                predited_label[i].argmax() + 1
                category = int(label[i]) + 1
                dic_total[category] = dic_total[category] + 1
                if int(predited_label[i].argmax()) + 1 == category:
                    dic_correct[category] = dic_correct[category] + 1
        print('_________________________\nAccuracy per category:')
        for each in dic_correct:
            category_accuracy = dic_correct[each] / dic_total[each]
            dic_accuracy[each] = category_accuracy
            print(f"Accuracy for {each} is {np.round(category_accuracy, 2)}")
        return dic_accuracy

    def create_model(self):
        self.load_datasets()

        EPOCHS = 10
        total_accu = None
        loss_tr = []
        loss_ev = []
        list_val_acc=[]
        highest_val_accuracy = 0
        ########################
        # 4. Define model:
        tokenizer = get_tokenizer('basic_english')

        num_class = 4
        # vocab_size = mat.shape[1]
        vocab_size = len(self.vectorizer.vocabulary_)
        emsize = 3

        ################################33
        # Define training of model

        self.model = TextClassificationModel(vocab_size, emsize, num_class, batch_size=self.batch_size)
        self.criterion = torch.nn.CrossEntropyLoss()
        # optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.2)
        self.optimizer = torch.optim.Adagrad(self.model.parameters(), lr=0.99)
        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1, gamma=0.1)

        for epoch in range(1, EPOCHS + 1):
            epoch_start_time = time.time()
            loss_training = self.train(epoch)
            accu_val, loss_evaluation = self.evaluate()
            list_val_acc.append(accu_val)
            if accu_val > highest_val_accuracy:
                highest_val_accuracy = accu_val
            self.accuracy_per_category()
            print("Mean loss training: ", np.mean(loss_training))
            loss_tr.append(np.mean(loss_training))
            loss_ev.append(np.mean(loss_evaluation))
            print("Mean loss evaluation: ", np.mean(loss_evaluation) )
            if total_accu is not None and total_accu > accu_val:
              scheduler.step()
            else:
               total_accu = accu_val
            print(total_accu)
            print('-'* 59)
            print('| end of epoch {:3d} | time: {:5.2f}s | '
                  'current valid accuracy {:8.3f} '.format(epoch,
                                                   time.time() - epoch_start_time,
                                                   accu_val))
            print('- '*30)
            print('Highest validation accuracy so far: ', highest_val_accuracy)
            print('-' * 59)
        plt.plot(loss_tr,'--', loss_ev, '--')
        plt.legend(['training','validation'])
        plt.title("Training and validation loss")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.xticks(np.arange(len(loss_tr)))
        plt.show()
        plt.savefig('./data_processing/topic modelling/exported_graphs/loss.png')
        # os.getcwd()
        print("Highest accuracy: ",np.round(np.max(list_val_acc),4), " from itteration ", np.argmax(list_val_acc))
        self.accuracy_per_category()


    ##########################3
    # Evaluate model

    def predict(self,text):
        with torch.no_grad():
            text = torch.tensor(text)
            text = text.type(torch.float)
            # text = text.unsqueeze(0)
            output = self.model(text)
            # print(output[0])
            return output[0].argmax(1)

    # print(target[:-BATCH_SIZE])
    # print(training_data.sentence.iloc[-1], target_labels[training_data.label.iloc[-1]])

#
# m = Model()
# m.create_model()
#



