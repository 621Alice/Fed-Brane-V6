import time
from vantage6.tools.util import info, warn
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import time

import torch
import torch.nn.functional as F
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torchvision import transforms
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from collections import OrderedDict, Counter
import numpy as np
import random
import nltk
import re
from tqdm.autonotebook import tqdm
from sklearn.metrics import accuracy_score

#split training file for multiple nodes
#non-iid /iid data? not concerned
#no local dp (dependencies not match)
def master(client, data,ids):
    """Combine partials to global model
    """

    info('Collecting participating organizations')

    # Collect all organization that participate in this collaboration.
    # These organizations will receive the task to compute the partial.
    # organizations = client.get_organizations_in_my_collaboration()
    # ids = [organization.get("id") for organization in organizations]

    # Determine the device to train on
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Train without federated averaging
    info('Train_test')
    task = client.create_new_task(
        input_={
            'method': 'train_test',
            'kwargs': {
                'model': "",
                'parameters': "",
                'device': device,
                'return_params': True,
                'epochs': 1,
                'if_test': False
            }
        }, organization_ids=ids
    )

    info("Waiting for results")
    task_id = task.get("id")
    task = client.get_task(task_id)
    while not task.get("complete"):
        task = client.get_task(task_id)
        info("Waiting for results")
        time.sleep(1)

    info("Obtaining results")

    results = client.get_results(task_id=task.get("id"))

    global_sum = 0
    global_count = 0

    for output in results:
        # print(len(output))
        global_sum += output["params"]
        global_count += len(global_sum)

    averaged_parameters = global_sum / global_count

    # in order to not have the optimizer see the new parameters as a non-leaf tensor, .clone().detach() needs
    # to be applied in order to turn turn "grad_fn=<DivBackward0>" into "grad_fn=True"
    averaged_parameters = [averaged_parameters.clone().detach()]

    info('Federated averaging w/ averaged_parameters')
    task = client.create_new_task(
        input_={
            'method': 'train_test',
            'kwargs': {
                'model': output['model'],
                'parameters': averaged_parameters,
                'device': device,
                'return_params': True,
                'epochs': 1,
                'if_test': True
            }
        },
        organization_ids=ids
    )

    info("Waiting for results")
    task_id = task.get("id")
    task = client.get_task(task_id)
    while not task.get("complete"):
        task = client.get_task(task_id)
        info("Waiting for results")
        time.sleep(1)

    info("Obtaining results")

    results = client.get_results(task_id=task.get("id"))
    for output in results:
        acc = output["test_accuracy"]
    return acc


def RPC_train_test(data, model, parameters, device, return_params, epochs, if_test):
    """Compute the average partial
       Code referenced to https://sofiadutta.github.io/datascience-ipynbs/pytorch/Sentiment-Analysis-using-PyTorch.html
    """
    # nltk.download('punkt')
    # nltk.download('stopwords')
    # en_stops = set(stopwords.words('english'))

    en_stops = {'hasn', 'nor', 'because', 'for', "you've", 'an', 'until', 'his', 'more', 'on', 'some',
                "wasn't", 'against', 'couldn', 'yourself', 'their', 'each', 'don', "needn't", 'isn', 'here',
                'under', 'hers', 'am', 't', 'o', 'wasn', 'yours', 'once', 'there', 'into', 'mightn', 'from',
                's', 'i', 'too', "you'll", 'be', 'and', "should've", "hasn't", 'ourselves', 'll', 'will',
                'didn', 'weren', 'aren', 'won', 'our', 'only', 'me', 'haven', 'now', 'what', 'than', "it's",
                'that', 'about', 'below',"that'll", 'then', 'where', 'so', "don't", 'he', 'mustn', 'in', 'a',
                'of', 'up', 'with', "aren't",'shan', 've', 'between', 'had', 'few', 'my', 'have', "you'd", 'while',
                'm', 'was', 'were', 'we', 'other',  'how','during', "you're", 'herself', 'its', "shan't", 'you',
                'himself', 'at', 'd', 'before', 'those', 'them', 'again','or', 'to', 'theirs', 'being', 'off',
                'needn', 'down', 'as', 'him', 'her', 'shouldn', 'which', 'not', 'whom', 'if',"she's", 'is', 'been',
                'through', 'just', "mightn't", "won't", "isn't", 'but', 'this', 'ours',"didn't", 'all', 'who',
                'does', 'out', 'over', 'after', 'it', 'myself', 'y', 'yourselves', 'themselves', 'most', 'ain', 'has',
                'are', 'own', "doesn't", 'having', "haven't", 'ma', 'no', 'any', 'why', 'above', "weren't", 'wouldn',
                'these', 'the', 'they', 'further', "couldn't", 'doing', 'can','do', 'when', 're', 'by', 'both', 'same',
                "wouldn't", 'she', 'such',"mustn't", 'did', "hadn't", 'your', 'itself', 'doesn', 'very', "shouldn't", 'hadn', 'should'}



    trainset_percent = 0.7
    B = 24  # batch size
    data_size = 35000
    print(int(data_size * trainset_percent))

    t = data[:data_size]
    # randomly shuffle the rows
    # t = t.sample(frac=1).reset_index(drop=True)
    print(t[:4])

    TEXT = t['Text']
    LABEL = t['Label']
    print(LABEL[0])
    print(TEXT[0])

    X = []
    for text in TEXT:
        # remove HTML, url and "@"
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"www\S+", "", text)
        text = re.sub(r"@\S+", "", text)
        # make lower case
        text = text.lower()
        # split string into words
        # list_of_words = word_tokenize(text)
        list_of_words = text.split()
        # remove punctuation and empty strings
        list_of_words_without_punctuation = [
            ''.join(this_char for this_char in this_string if (this_char in string.ascii_lowercase)) for this_string in
            list_of_words]
        list_of_words_without_punctuation = list(filter(None, list_of_words_without_punctuation))
        # remove stopwords
        filtered_word_list = [w for w in list_of_words_without_punctuation if w not in en_stops]
        X.append(' '.join(filtered_word_list))
    # print(X[0])

    vocab = Counter()
    # print(vocab)
    for x in X:
        vocab.update(x.split())
    # print(vocab)

    word_list = sorted(vocab, key=vocab.get, reverse=True)
    vocab_to_int = {word: idx + 1 for idx, word in enumerate(word_list)}
    int_to_vocab = {idx: word for word, idx in vocab_to_int.items()}

    train_label = LABEL[0:int(data_size * trainset_percent)]
    train_text = X[0:int(data_size * trainset_percent)]
    test_label = LABEL[int(data_size * trainset_percent):data_size]
    test_text = X[int(data_size * trainset_percent):data_size]
    train_dict = {'neg': [], 'pos': []}
    test_dict = {'neg': [], 'pos': []}

    for idx, l in enumerate(train_label):
        if l == 0:
            train_dict['neg'].append(train_text[idx])
        elif l == 4:
            train_dict['pos'].append(train_text[idx])

    print(train_dict['neg'][0])
    print(train_dict['pos'][0])
    print(len(train_dict))
    print(len(train_dict['neg']))
    print(len(train_dict['pos']))

    for idx, l in enumerate(test_label):
        if l == 0:
            test_dict['neg'].append(test_text[idx])
        elif l == 4:
            test_dict['pos'].append(test_text[idx])

    print(test_dict['neg'][0])
    print(test_dict['pos'][0])
    print(len(test_dict))
    print(len(test_dict['neg']))
    print(len(test_dict['pos']))

    train_dataset = SentimentDataset(train_dict, vocab, vocab_to_int)
    test_dataset = SentimentDataset(test_dict, vocab, vocab_to_int)

    train_loader = DataLoader(train_dataset, batch_size=B, shuffle=True, collate_fn=pad_and_pack)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=pad_and_pack)
    print(len(train_loader))

    D = 32
    alphabet_size = len(vocab) + 1
    hidden_nodes = 64
    classes = len(train_dataset.labels)

    loss_func = nn.CrossEntropyLoss()

    if if_test:
        #Next training with new model and averaged params and test
        optimizer = torch.optim.AdamW(parameters, lr=0.001 * B)

    else:
        #first training without FL
        # initialize model
        model = nn.Sequential(
            EmbeddingPackable(nn.Embedding(alphabet_size, D)),  # (B, T) -> (B, T, D)
            nn.LSTM(D, hidden_nodes, num_layers=3, batch_first=True, bidirectional=True),
            # (B, T, D) -> ( (B,T,D) , (S, B, D)  )
            LastTimeStep(rnn_layers=3, bidirectional=True),
            # We need to take the RNN output and reduce it to one item, (B, D)
            nn.Linear(hidden_nodes * 2, classes),  # (B, D) -> (B, classes)
        )
        print(model.eval())
        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001 * B)

    for epoch in tqdm(range(epochs), desc="Epoch", disable=False):
        model = model.train()
        running_loss = 0.0

        y_true = []
        y_pred = []

        for inputs, labels in tqdm(train_loader, desc="Train Batch", leave=False, disable=False):
            inputs = inputs.to(device)
            labels = labels.to(device)

            batch_size = labels.shape[0]
            optimizer.zero_grad()
            y_hat = model(inputs)
            loss = loss_func(y_hat, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_size
            labels = labels.detach().cpu().numpy()
            y_hat = y_hat.detach().cpu().numpy()
            for i in range(batch_size):
                y_true.append(labels[i])
                y_pred.append(y_hat[i, :])

        y_pred = np.asarray(y_pred)
        if y_pred.shape[1] > 1:
            y_pred = np.argmax(y_pred, axis=1)

        train_acc = accuracy_score(y_true, y_pred)
        print(f'Train accuracy for epoch {epoch + 1}:{train_acc * 100}%')

    if if_test:
        with torch.no_grad():
            correct = 0
            for X_test, y_test in test_loader:
                X_test = X_test.to(device)
                y_test = y_test.to(device)
                y_val = model(X_test)
                predicted = torch.max(y_val, 1)[1]
                correct += (predicted == y_test).sum()
        print(
            f'Test accuracy: {correct.item()}/{len(test_dataset)} = {correct.item() * 100 / (len(test_dataset)):7.3f}%')
        test_acc = correct.item() * 100 / (len(test_dataset))
    else:
        test_acc = 0

    if return_params:
        for parameters in list(model.parameters()):
            return {'params': parameters,
                    'model': model,
                    'test_accuracy': test_acc}


def pad_and_pack(batch):
    input_tensors = []
    labels = []
    lengths = []
    for x, y in batch:

        if x.shape[0] > 0:
            input_tensors.append(x)
            labels.append(y)
            lengths.append(x.shape[0])
    if len(input_tensors[0].shape) == 1:
        x_padded = torch.nn.utils.rnn.pad_sequence(input_tensors, batch_first=False)
    else:
        raise Exception('Current implementation only supports (T) shaped data')

    x_packed = torch.nn.utils.rnn.pack_padded_sequence(x_padded, lengths, batch_first=False, enforce_sorted=False)

    y_batched = torch.as_tensor(labels, dtype=torch.long)

    return x_packed, y_batched


class SentimentDataset():
    def __init__(self, dict, alphabet, vocab_to_int):

        self.data = dict
        self.labels = [x for x in dict.keys()]
        self.alphabet = alphabet
        self.vocab_to_int = vocab_to_int

    def __len__(self):
        return sum([len(x) for x in self.data.values()])

    def __getitem__(self, idx):
        label = 0
        while idx >= len(self.data[self.labels[label]]):
            idx -= len(self.data[self.labels[label]])
            label += 1
        Text = self.data[self.labels[label]][idx]

        label_vec = torch.zeros((1), dtype=torch.long)
        label_vec[0] = label
        return self.Text2InputVec(Text), label

    def Text2InputVec(self, text):
        T = len(text)
        text_vec = torch.zeros((T), dtype=torch.long)
        for pos, word in enumerate(text.split()):
            if word not in self.vocab_to_int.keys():
                text_vec[pos] = 0
            else:
                text_vec[pos] = self.vocab_to_int[word]

        return text_vec



class EmbeddingPackable(nn.Module):

    def __init__(self, embd_layer):
        super(EmbeddingPackable, self).__init__()
        self.embd_layer = embd_layer


    def forward(self, input):
        if type(input) == torch.nn.utils.rnn.PackedSequence:
            # We need to unpack the input,
            sequences, lengths = torch.nn.utils.rnn.pad_packed_sequence(input.cpu(), batch_first=True)
            # Embed it
            sequences = self.embd_layer(sequences.to(input.data.device))
            # And pack it into a new sequence
            return torch.nn.utils.rnn.pack_padded_sequence(sequences, lengths.to(input.data.device),
                                                           batch_first=True, enforce_sorted=False)
        else:  # apply to normal data
            return self.embd_layer(input)


class LastTimeStep(nn.Module):

    def __init__(self, rnn_layers=1, bidirectional=False):
        super(LastTimeStep, self).__init__()
        self.rnn_layers = rnn_layers
        if bidirectional:
            self.num_driections = 2
        else:
            self.num_driections = 1

    def forward(self, input):
        # Result is either a tupe (out, h_t)
        # or a tuple (out, (h_t, c_t))
        rnn_output = input[0]
        last_step = input[1]
        if (type(last_step) == tuple):
            last_step = last_step[0]
        batch_size = last_step.shape[1]  # per docs, shape is: '(num_layers * num_directions, batch, hidden_size)'

        last_step = last_step.view(self.rnn_layers, self.num_driections, batch_size, -1)
        # We want the last layer's results
        last_step = last_step[self.rnn_layers - 1]
        # Re order so batch comes first
        last_step = last_step.permute(1, 0, 2)
        # Finally, flatten the last two dimensions into one
        return last_step.reshape(batch_size, -1)
