# -*- coding: utf-8 -*-
"""Lab6.ipynb """

# !pip install pytorch-crf
# !pip install datasets
# !pip install sklearn
# !pip install transformers

# Commented out IPython magic to ensure Python compatibility.
# %reload_ext autoreload
# %autoreload 2
# %matplotlib inline

import io
from math import log
import os
import pickle
from numpy import array
from numpy import argmax
import torch
import random
from math import log
from numpy import array
from numpy import argmax
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.optim import Adam
from torchcrf import CRF
from torch.optim.lr_scheduler import ExponentialLR, CyclicLR
from typing import List, Tuple, AnyStr
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
from copy import deepcopy
from datasets import load_dataset, load_metric
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import transformers
from transformers import AutoTokenizer, AdamW
from datasets import DatasetDict
from dataclasses import dataclass
import random
import time
import datetime
import sys
import math


def enforce_reproducibility(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
enforce_reproducibility()

@dataclass
class DataPoint:
    """Class that represents a datapoint"""

    qst: str
    qstTokenized: list
    ans: str
    ansTokenized: list  # answer tokenizer
    fullText: str  # raw full text
    fullTextTokenized: str  # full text tokenizer
    prompt: list  # question + full text, tokenized
    lbl: str  # TODO maybe bool instead?


def loadModel(mn):
    appendToLogFile(f"Loading model: {mn}")
    return transformers.AutoModelForTokenClassification.from_pretrained(
        mn,
        output_hidden_states=True,
        num_labels=len(getLabels()),
    ).to(device)


def loadTokenizer(mn):
    appendToLogFile(f"Loading tokenizer: {mn}")
    return AutoTokenizer.from_pretrained(mn)


def loadTyDiQaDataset():
    return load_dataset("copenlu/answerable_tydiqa")


def appendToLogFile(text):
    """Appends text to a log file"""
    with open(logName, "a") as f:
        timeStamp = datetime.datetime.now().time()
        f.write(f"{timeStamp}: {text}")
        # Check if text string ends with a new line, if not then add one. Beware of empty text strings.
        if text and text[-1] != "\n":
            f.write("\n")


def printAndLog(text):
    """Prints and logs text"""
    print(text)
    appendToLogFile(text)


def idxToLabel(idx):
    labels = {0: "O", 1: "B", 2: "I", -100: "IGN"}
    return labels[idx]


def getLabels():
    return {"O": 0, "B": 1, "I": 2, "IGN": -100}


def getLogRegLabels():
    return {"UNANS": 0, "ANS": 1}


def idxToLabelLogReg():
    return {0: "UNANS", 1: "ANS"}


def calcClassWeights(target):
    """Calculate inverse class weights for a given target"""
    # Collect number of occurences
    target = np.array(target)
    weights = np.bincount(target)
    printAndLog(f"Label counts: O: {weights[0]}, B: {weights[1]}, I: {weights[2]}")

    # Calculate inverse class weights
    weights = 1 / weights
    weights /= weights.sum()
    weights = np.append(weights, 0)  # append 0 weight for the "IGN" label
    return weights


def getStartEndIndices(l, sl):
    if len(sl) == 0:
        return (-1, -1)
    sll = len(sl)
    for ind in (i for i, e in enumerate(l) if e == sl[0]):
        if l[ind : ind + sll] == sl:
            return (ind, ind + sll - 1)
    return (-1, -1)


def parseData(dataSet: DatasetDict, language: str, dsType: str):
    """Parses the dataset into a list of data structures
    Args:
        dataSet (DatasetDict): Dataset
        language (str): Language for which to get the datapoints
        dsType (str): Type of the dataset (train, validation, test)

    Returns:
        np.ndarray: List of DataPoint entries
    """
    from collections import Counter

    appendToLogFile(f"-- Data Parsing {language.upper()}; Type: {dsType.upper()}--")

    # Convert ds to a panda and take out all the goodies
    ds = dataSet[dsType].to_pandas()
    ds = ds.loc[ds["language"] == language]  # Filter out the language
    dsQ = ds["question_text"].values  # Questions
    dsAns = ds["annotations"].values  # Answer stuff
    dsAnsStart = [i["answer_start"][0] for i in dsAns]  # Char idx answer start
    dsAns = [i["answer_text"][0] for i in dsAns]  # Raw shortest answer text
    dsFullText = ds["document_plaintext"].values  # Full text
    assert len(dsQ) == len(dsAns), "Number of questions and answers is not the same"

    # Prepare utility variables
    cnt = 0
    fails = 0
    data = np.array([], dtype=DataPoint)
    notAnsCnt = 0
    labelCount = []
    # Pack the data in a datastructure and save it to an array
    for i, qst in tqdm(enumerate(dsQ)):
        if not torch.cuda.is_available() and i >= 25:
            break

        context = dsFullText[i]
        qstTkn = TOKENIZER(qst, return_tensors="pt")
        ansTkn = TOKENIZER(dsAns[i], return_tensors="pt")
        contextTkn = TOKENIZER(context, return_tensors="pt")
        prompt = TOKENIZER(qst + "\n" + context, return_tensors="pt")

        # if prompt length is too long, skip
        if len(prompt["input_ids"][0]) > 512:
            cnt += 1
            continue

        # Convert the tensor input ids back to tokens to map out start-end indices
        qstTkn = TOKENIZER.convert_ids_to_tokens(qstTkn["input_ids"][0])[
            :-1
        ]  # Remove [SEP] token
        prompt_txt = TOKENIZER.convert_ids_to_tokens(prompt["input_ids"][0])
        ansTknTxt = TOKENIZER.convert_ids_to_tokens(
            ansTkn["input_ids"][0], skip_special_tokens=True
        )

        # Find the start and end indices of the answer tokens in the context tokens
        startIdx, endIdx = getStartEndIndices(prompt_txt, ansTknTxt)

        # Assert correct start-end indices, if theres a fail then skip this datapoint
        try:
            for i, ans in zip(range(startIdx, endIdx + 1), ansTknTxt):
                if prompt_txt[i] != ans:
                    raise Exception("Answer extraction failed")
        except:
            fails += 1
            # if language is english, then log the failed datapoint and dstype is training
            if language == "english" and dsType == "train" and fails < 10:
                appendToLogFile(
                    f"Failed Answer extraction: {ansTknTxt}; Prompt: {prompt_txt}"
                )
            continue

        # Assign the labels. They should always end up in a continuous sequence
        labels = []
        j = 0  # index of the answer token
        for i, tkn in enumerate(prompt_txt):
            # check if we are going over the question
            if i < len(qstTkn):
                labels.append(getLabels()["IGN"])
            # We've hit the first answer token
            elif i == startIdx:
                if prompt_txt[i] == ansTknTxt[j]:  # last layer of error-checking
                    labels.append(getLabels()["B"])  # all is OK, append the label
                    j += 1
                else:
                    print("Error")
            # We're in the middle of the answer tokens
            elif i > startIdx and i <= endIdx:
                if prompt_txt[i] == ansTknTxt[j]:  # last layer of error-checking
                    labels.append(getLabels()["I"])  # all is OK, append the label
                    j += 1
                else:
                    print("Error")
            # All other tokens are O
            else:
                labels.append(getLabels()["O"])
        
        if getLabels()["B"] not in labels:
            notAnsCnt += 1

        # Create the data points
        labelCount = labelCount + labels  # keep track of the categories
        entry = DataPoint(qst, qstTkn, ans, ansTkn, None, None, prompt, labels)
        data = np.append(data, entry)

    # Print some stats
    printAndLog(f"Unanswerable questions: {notAnsCnt}")
    printAndLog(
        f"Balance of labels: {Counter(labelCount).keys()}:{Counter(labelCount).values()}"
    )
    if cnt > 0 or fails > 0:
        printAndLog(f"Entries skipped due to too long sequence length (>512): {cnt}")
        printAndLog(f"Failed to map answer and to context: {fails}")

    # Get class weights
    classWeights = None
    labelCount = np.array(labelCount)
    labelCount = np.delete(
        labelCount, np.where(labelCount == getLabels()["IGN"]), axis=0
    )  # remove IGN label from class weights
    if dsType == "train" and len(sys.argv) > 0:
        classWeights = calcClassWeights(labelCount)

    # Return output
    output = data.ravel()
    printAndLog(f"Final length: {len(output)} \n\n")
    return output, classWeights


def getDataLoader(dataset, bs, shuffle, n_workers=0):
    """Returns a dataloader object"""
    return DataLoader(
        dataset,
        batch_size=bs,
        shuffle=shuffle,
        collate_fn=collate_batch_bilstm,
        num_workers=n_workers,
    )


def collate_batch_bilstm(input_data) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns a sequence of labels instead of a single label as in text classification."""

    input_ids = [entry.prompt for entry in input_data]
    seq_lens = [len(entry["input_ids"][0]) for entry in input_ids]
    labels = [entry.lbl for entry in input_data]
    attention_mask = [entry["attention_mask"][0].numpy() for entry in input_ids]
    input_ids_real = np.array(
        [i["input_ids"][0].numpy() for i in input_ids], dtype=object
    )

    attention_mask = np.squeeze(attention_mask)
    input_ids_real = np.squeeze(input_ids_real)
    max_length = max(seq_lens)

    # Pad the input ids and attention mask
    inpIds = np.zeros((len(input_ids_real), max_length), dtype=np.int64)
    attMask = np.zeros((len(attention_mask), max_length), dtype=np.int64)
    for i, (seq, seq_len) in enumerate(zip(input_ids_real, seq_lens)):
        inpIds[i, :seq_len] = seq
        attMask[i, :seq_len] = attention_mask[i]

    labels = np.array(
        [np.array((i + [getLabels()["IGN"]] * (max_length - len(i)))) for i in labels]
    )        


    assert all(len(i) == max_length for i in labels)
    output = {
        "input_ids": torch.tensor(inpIds).to(device),
        "attention_mask": torch.tensor(attMask).to(device),
        "labels": torch.tensor(labels).to(device),
    }
    return output


def train(
    model: nn.Module,
    train_dl: DataLoader,
    val_dl: DataLoader,
    optimizer: torch.optim.Optimizer,
    n_epochs: int,
    device: torch.device,
    modelName,
    scheduler,
):
    """
    The main training loop which will optimize a given model on a given dataset
    :param model: The model being optimized
    :param train_dl: The training dataset
    :param valid_dl: A validation dataset
    :param optimizer: The optimizer used to update the model parameters
    :param n_epochs: Number of epochs to train for
    :param device: The device to train on
    :return: (model, losses) The best model and the losses per iteration
    """
    # import logistic regression
    from sklearn.linear_model import LogisticRegression

    # Keep track of the loss and best accuracy
    losses = []
    best_f1_sl = 0.0
    best_f1_bc = 0.0
    trainDl = train_dl[0]
    trainLang = train_dl[1]
    valSets = []
    criterion = torch.nn.CrossEntropyLoss(
        weight=torch.FloatTensor(c_weights).to(device)
    )
    learning_rates = []
    logReg = LogisticRegression(penalty="l2", solver='saga', max_iter=1500)

    # Iterate through epochs
    for ep in range(n_epochs):
        model.train()
        loss_epoch = []
        appendToLogFile(
            f"Model: {modelName}; Language: {trainLang}; Epoch {ep+1}/{n_epochs}"
        )

        Xval = []
        yval = []

        # Iterate through each batch in the dataloader
        for inputs in tqdm(trainDl):
            # zero them out
            optimizer.zero_grad()

            # Pass the inputs through the model, get the current loss and logits
            labels = inputs["labels"]
            output = model(**inputs)
            logits = output["logits"].swapaxes(1, 2)

            loss = criterion(logits, labels)
            losses.append(loss)
            loss_epoch.append(loss)

            # Calculate all of the gradients and weight updates for the model
            loss.backward()
            optimizer.step()
            if scheduler != None:
                scheduler.step()
                #learning_rates.append(scheduler.get_last_lr()[0])

            last_hid_lay = torch.stack(list(output["hidden_states"]), dim=0)[-1][:, :, :]
            Xval.extend(
                torch.sum(last_hid_lay, dim=1).detach().clone().cpu().numpy().tolist()
            )
            labels = labels.detach().clone().cpu().numpy().tolist()
            # if label contains "B" or "I" then extend yval with "ANS", else extend with "UNANS"
            for entry in labels:
                if getLabels()["B"] in entry or getLabels()["I"] in entry:
                    yval.append(getLogRegLabels()["ANS"])
                else:
                    yval.append(getLogRegLabels()["UNANS"])

        # train and evaluate log.reg
        hs = np.squeeze(np.asarray(Xval))
        labels = np.squeeze(np.asarray(yval))
        logReg.fit(Xval, yval)

        # Perform inline evaluation at the end of the epoch
        combinedF1_sl = 0
        combinedF1_bc = 0
        for vd in val_dl:

            printAndLog(f"Evaluating Language: {vd[1]}")
            printAndLog(("-" * 10))

            F1_sl, F1_bc = eval(model, logReg, vd[0])
            combinedF1_sl += F1_sl * F1_sl
            printAndLog(
                f"Validation Seq.Label F1: {F1_sl}; Log.Reg F1: {F1_bc}; train loss: {sum(loss_epoch) / len(loss_epoch)}; Language: {vd[1]} \n\n"
            )

            combinedF1_bc += F1_bc * F1_bc
            # Enter binary classification code HERE

        # Calculated combined F1 score for sequence labeling
        combinedF1_sl = math.sqrt(combinedF1_sl / len(val_dl))
        printAndLog(
            f"Combined F1 SeqLab: {combinedF1_sl}; train loss: {sum(loss_epoch) / len(loss_epoch)}"
        )
        if combinedF1_sl > best_f1_sl:
            torch.save(model.state_dict(), modelName)
            best_f1_sl = combinedF1_sl

        # Calculated combined F1 score for binary classification
        combinedF1_bc = math.sqrt(combinedF1_bc / len(val_dl))
        printAndLog(
            f"Combined F1 LogReg: {combinedF1_bc}; train loss: {sum(loss_epoch) / len(loss_epoch)} \n\n"
        )
        if combinedF1_bc > best_f1_bc:
            # use pickle to save the model
            with open(modelName + "_logReg.pck", "wb") as f:
                pickle.dump(logReg, f)
                best_f1_bc = combinedF1_bc

    printAndLog(f"Learning rates: {learning_rates}")
    return losses


def eval(model: nn.Module, logReg, valid_dl: DataLoader):
    """
    Evaluates the model on the given dataset
    :param model: The model under evaluation
    :param valid_dl: A `DataLoader` reading validation data
    :return: The accuracy of the model on the dataset
    """
    # VERY IMPORTANT: Put model in "eval" mode
    model.eval()
    labels_all = []
    preds_all = []

    predsLogRegr = []
    logRegrLabels = []

    # ALSO IMPORTANT: Don't accumulate gradients during this process
    with torch.no_grad():
        for inputs in tqdm(valid_dl, desc="Evaluation"):
            # Pass the inputs through the model, get the current loss and logits
            labels = inputs["labels"]
            output = model(**inputs)

            # Get all indices of elements that are not class 3
            # match all of those predictions with labels indices
            labels = labels.detach().cpu().numpy()
            preds = torch.argmax(output.logits, dim=-1).detach().cpu().numpy()
            
            # fetch all indices where labels are IGN and delete the entries
            indices = np.where(labels == getLabels()["IGN"])
            labels = np.delete(labels, indices, axis=1)
            preds = np.delete(preds, indices, axis=1)
            
            preds_all.extend(preds)
            labels_all.extend(labels)

            # Prepare data for logistic regression
            last_hid_lay = torch.stack(list(output["hidden_states"]), dim=0)[-1][:, :, :]
            last_hid_lay = (torch.sum(last_hid_lay, dim=1).detach().clone().cpu().numpy().tolist())
            
            for entry in labels:
                if getLabels()["B"] in entry:
                    logRegrLabels.append(getLogRegLabels()["ANS"])
                else:
                    logRegrLabels.append(getLogRegLabels()["UNANS"])

            #Evaluate binary classification
            preds = logReg.predict(last_hid_lay)
            predsLogRegr.extend(preds)

    # Perform analysis and logging
    # flatten labels_all and preds_all
    labels_all = [item for sublist in labels_all for item in sublist]
    preds_all = [item for sublist in preds_all for item in sublist]
    P, R, F1_seqLabel, _ = precision_recall_fscore_support(
        labels_all, preds_all, average="macro"
    )
    P, R, F1_binClass, _ = precision_recall_fscore_support(
        logRegrLabels, predsLogRegr, average="macro"
    )

    printAndLog(f"OBI: \n {np.array2string(confusion_matrix(labels_all, preds_all))}")
    printAndLog(f"OBI: \n {classification_report(labels_all, preds_all)}")
    appendToLogFile("-" * 10)
    printAndLog(f"LR: \n {np.array2string(confusion_matrix(logRegrLabels, predsLogRegr))}")
    printAndLog(f"LR: \n {classification_report(logRegrLabels, predsLogRegr)}")

    return F1_seqLabel, F1_binClass

timeStamp = time.strftime("%Y%m%d-%H%M%S")
currFileLoc = ""
logName = os.path.join(currFileLoc, f"l6_log_{timeStamp}.txt")
with open(logName, "w") as f:
    f.write("")

appendToLogFile("Start of log file \n")
appendToLogFile(f"Using CUDA: {torch.cuda.is_available()} \n\n")

# Set constants
DEBUG = False


# Training hyperparameters
device = (torch.device("cpu"), torch.device("cuda"))[torch.cuda.is_available()]
dropout_prob = 0.25
batch_size = 32
lr = 0.00005
n_epochs = 40
appendToLogFile(f"dropout_prob: {dropout_prob}; batch_size: {batch_size}; lr: {lr}; n_epochs: {n_epochs}")

# Load and parse the dataset
languages = ["english", "finnish", "japanese"]


modelNames = ["bert-base-multilingual-cased", "xlm-roberta-base"]
ds = loadTyDiQaDataset()

for mn in modelNames:
    TOKENIZER = loadTokenizer(mn)
    
    # create the validation sets
    valSetOne, _ = parseData(ds, languages[0], "validation")
    valSetTwo, _ = parseData(ds, languages[1], "validation")
    valSetThree, _ = parseData(ds, languages[2], "validation")
    valDlOne = getDataLoader(valSetOne, bs=batch_size, shuffle=True)
    valDlTwo = getDataLoader(valSetTwo, bs=batch_size, shuffle=True)
    valDlThree = getDataLoader(valSetThree, bs=batch_size, shuffle=True)

    valDl = [(valDlOne, "english"), (valDlTwo, "finnish"), (valDlThree, "japanese")]


    for language in languages:
        torch.cuda.empty_cache()
        modelName = f"lab6_{mn}_{language}.pt"

        # Load the training dataset, and two validation sets of the other two languages
        trainSet, c_weights = parseData(ds, language, "train")
        appendToLogFile(f"Language: {language}; Class weights: {c_weights}")

        trainDl = getDataLoader(trainSet, bs=batch_size, shuffle=True)
        trainDl = (trainDl, language)

        appendToLogFile(f"Training model: {modelName}")
        model = loadModel(mn)
        optimizer = AdamW(model.parameters(), lr=lr)
        scheduler = CyclicLR(
            optimizer,
            base_lr=0.0,
            max_lr=lr,
            step_size_up=1,
            step_size_down=(len(trainDl) * n_epochs),
            cycle_momentum=False,
        )
        train(model, trainDl, valDl, optimizer, n_epochs, device, modelName, scheduler)

appendToLogFile("Finished successfully!")