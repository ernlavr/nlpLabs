import sys
from typing import List
import matplotlib.pyplot as plt
import os
from dataclasses import dataclass
import pandas as pd
import numpy as np
import seaborn as sn

# get filename from second argument
#fileName = sys.argv[1]


def resetState():
    language = ""
    modelName = ""
    classWeights = []
    f1 = []
    loss = []

fig, ax = plt.subplots()
def addToPlot(modelName, f1, loss):
    ax.plot(f1, label=modelName)
    ax.plot(loss, label=modelName)

def showPlot():
    ax.legend()
    plt.show()


# get the current file's directory as a string
dir_path = os.path.dirname(os.path.realpath(__file__))
filePath = os.path.join(dir_path, "lab6_results.txt")



@dataclass
class DataPoint():
    name : list
    classWeights : list
    cwDivisor : list
    res : list # f1, logRegF1, loss
    accuracy : list
    epoch : list

def plotResults(datapoint : dict, task : str):
    """ Plots the results of datapoint dictionary. Keys are string denoting languages
    and values are DataPoint classes.
    The data is organized such that "f1" and "loss" lists each contain 75 entries that denote each epoch
    "f1" and "loss" lists can be processed in pairs of two, i.e. [0] and [1] index,
    [0] index contains a basic-model data and [1] contains a modified model with unity weights,e
    [2] contains a basic-model and [3] contains a modified model with modified unity weights,
    same pattern follows for [5] and [6] entries

    A single plot should contain results of all three languages per model type and per weight type.
    This should result in total of six plots each containing data of three languages.
    """

    # model sequence
     
    if task == "Sequence Labeling":
        tmp = "SeqLab"
    else:
        tmp = "BinCl"
    confusionMatrix = {"english_xlm" : [], "finnish_xlm" : [], "japanese_xlm" : [], \
                        "english_bert" : [], "finnish_bert" : [], "japanese_bert" : []}
    # create a plot for each language
    for model in datapoint:
        # create an empty lineplot canvas, x axis should be epochs 1 to 75, there should be two y axis for f1 and loss
        fig, ax = plt.subplots()
        totalEpochs = 40

        # get the name of the model
        modelName = datapoint[model].name[0][0]
        

        eng = []
        fin = []
        jap = []
        for epochRes, epochAcc in zip(datapoint[model].res, datapoint[model].accuracy):
            for res, acc in zip(epochRes, epochAcc):
                if res[3] == "english":
                    eng.append((res, acc))
                if res[3] == "finnish":
                    fin.append((res, acc))
                if res[3] == "japanese":
                    jap.append((res, acc))
        
        res = [eng, fin, jap]
        for entry in res:
            # entry[0] is f1, [1] is f1LogReg, [2] is loss and [3] is language
            # plot f1 against loss
            
            
            if task == "Sequence Labeling":
                f1 = [i[0][0] for i in entry]
            elif task == "Binary Classification":
                f1 = [i[0][1] for i in entry]
                acc = [i[1] for i in entry]
            else:
                raise Exception("Incorrect plotting mode")

            loss = [i[0][2] for i in entry]
            lng = [i[0][3] for i in entry]

            # plot the f1 and loss on two different y axis and denote epochs on x axis
            # label should denote 
            ax.plot(f1, label=f"F1_{lng[0]}") 
            # highlight hte maximum F1 value and the epoch it occured
            maxF1 = max(f1)
            maxF1Epoch = f1.index(maxF1)
            # Print the best epoch and the corresponding language and model
            #print(f"Best F1 for {modelName} with {lng} is {maxF1} at epoch {maxF1Epoch}")

            ax.scatter(maxF1Epoch, maxF1, label=f"Max F1_{model}")
            # write down the maxF1 score in a plot description
            # get the language from the modelName

            ax.text(maxF1Epoch, maxF1, f"{maxF1:.2f}")

            # add last epoch to the confusion matrix
            confusionEntry = model.split("-")[0]
            if task == "Binary Classification":
                confusionMatrix[confusionEntry].append(acc[-1])    
            else:
                confusionMatrix[confusionEntry].append(f1[-1])


        if "english" in modelName:
            loss = [i[0][2] for i in eng]
            lng = lng = "english"
        elif "finnish" in  modelName:
            loss = [i[0][2] for i in fin]
            lng = lng = "finnish"
        elif "japanese" in modelName:
            loss = [i[0][2] for i in jap]
            lng = "japanese"

        ax.plot(loss, label=f"Loss_{lng}")
        
        # highlight the minimum loss value and the epoch it occured
        minLoss = min(loss)
        minLossEpoch = loss.index(minLoss)
        ax.scatter(minLossEpoch, minLoss, label=f"Min Loss_{model}")

        # write down the minLoss score in a plot description
        ax.text(minLossEpoch, minLoss, f"{minLoss:.2f}")


        # add a legend to the plot
        ax.legend()

        # set the title to denote class weights and divisor
        ax.set_title(f"{task} F1 scores: {modelName[0]}")
        
        # zoom out the plot and place the legend outside the plot
        fig.set_size_inches(9, 4.5)
        # place the legend outside the plot
        ax.legend(bbox_to_anchor=(1.04, 1), loc='upper left', borderaxespad=0.)

        # annotate x axis label as epoch
        ax.set_xlabel("Epoch")
        # annotate y axis label as f1 and loss
        ax.set_ylabel("F1 and Loss")
        # shift the plot off-center
        ax.set_position([0.1, 0.1, 0.55, 0.85])

        # always show the last number on x axis 75
        ax.set_xticks(range(0, totalEpochs, 2))

        # save the plot to a file if cwDivisor is (1, 1)
        print(f"lab6_{tmp}_{model}.png")
        plt.savefig(f"lab6_{tmp}_{model}.png") 
    
    # make confusion matrix 6x6, pad first three element with zero at the end
    # and last three elements with zero at the beginning
    # this is done to make the matrix symmetric
    
    # confusionMatrix["english_xlm"].extend([0, 0, 0])
    # confusionMatrix["finnish_xlm"].extend([0, 0, 0])
    # confusionMatrix["japanese_xlm"].extend([0, 0, 0])
    # confusionMatrix["english_bert"] = [0, 0, 0] + confusionMatrix["english_bert"]
    # confusionMatrix["finnish_bert"] = [0, 0, 0] + confusionMatrix["finnish_bert"]
    # confusionMatrix["japanese_bert"] = [0, 0, 0] + confusionMatrix["japanese_bert"]
    

    array = np.array([confusionMatrix["english_xlm"], confusionMatrix["finnish_xlm"], confusionMatrix["japanese_xlm"]])
    df_cm = pd.DataFrame(array, index = [i for i in "Eng_XLM Fin_XLM Jap_XLM".split()],
                  columns = [i for i in "Eng_XLM Fin_XLM Jap_XLM".split()])
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True, vmin=0.3, vmax=.95)
    # Add title
    plt.title(f"{task} F1: XLM-R, E=75")
    if task == "Binary Classification":
        plt.title(f"{task} Accuracy: XLM-R, E=75")

    # save the plot to a file
    print(f"lab6_{tmp}_confusionMatrix.png")
    plt.savefig(F"lab6_{tmp}_xlm_confusionMatrix.png", bbox_inches='tight')


    array = np.array([confusionMatrix["english_bert"], confusionMatrix["finnish_bert"], confusionMatrix["japanese_bert"]])
    df_cm = pd.DataFrame(array, index = [i for i in "Eng_BERT Fin_BERT Jap_BERT".split()],
                  columns = [i for i in "Eng_BERT Fin_BERT Jap_BERT".split()])
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True,vmin=0.3 ,vmax=0.95)
    # Add title
    plt.title(f"{task} F1: M-BERT, E=75")
    if task == "Binary Classification":
        plt.title(f"{task} Accuracy: M-BERT E=75")

    # save the plot to a file
    print(f"lab6_{tmp}_confusionMatrix.png")
    plt.savefig(F"lab6_{tmp}_bert_confusionMatrix.png", bbox_inches='tight')



def main(): 
    language = ""
    currModel = ""
    epoch = 0
    results = {}

    # read and parse file called log
    with open(filePath) as f:
        # parse each line on a loop
        # create a line plot canvas
        parseAccuracy = None

        for cnt, line in enumerate(f):
            # remove timestamp and newline
            lineAcc = line
            line = line[17:-1]

            
            if "LR:" in line:
                parseAccuracy = True


            if "Loading tokenizer: " in line:
                tokenizer = line.split(": ")[1]
                results[f"english_{tokenizer}"] = DataPoint([], [], [], [], [], [])
                results[f"finnish_{tokenizer}"] = DataPoint([], [], [], [], [], [])
                results[f"japanese_{tokenizer}"] = DataPoint([], [], [], [],[], [])

            # check if line contains "Class weights [ 0.34900907 51.00340136  8.68520127]"
            if "Language: " in line:
                language = line.split(" ")[-1]
                # remove newline characters from language
                language = language.replace("\n", "")
                

            
            
            if "Starting training with CW divisor: " in line:
                # get substring between ( and )
                cwDivisor = line.split("(")[1].split(")")[0]
                cwDivisor = cwDivisor.split(", ")
                # parse cwDivisor to a float tuple
                cwDivisor = (float(cwDivisor[0]), float(cwDivisor[1]))
                
                # Hack for duplicate the entry for the second model for easier mapping during plotting
                results[resKey].cwDivisor.append(cwDivisor)
                results[resKey].cwDivisor.append(cwDivisor)

            if "New class weights" in line:
                # get substring between [ and ]
                classWeights = line.split("[")[1].split("]")[0]
                # split the string by "," and parse it to a float tuple
                classWeights = classWeights.split(", ")
                # round the floats to fifth digit
                classWeights = (round(float(classWeights[0]), 5), \
                                round(float(classWeights[1]), 5), \
                                round(float(classWeights[2]), 5) \
                                )
                
                # Hack for duplicate the entry for the second model for easier mapping during plotting
                results[resKey].classWeights.append(classWeights)
                results[resKey].classWeights.append(classWeights) 


            # Check if line contains "Model"
            if "Model: " in line:
                # locate "Language: " within the line and get the language
                line = line.replace(" ", "")
                components = line.split(";")
                language = components[1].split(":")[1]

                resKey = f"{language}_{tokenizer}"
                # extract the substring between : and ;
                modelName = components[0].split(":")[1]
                
                # check if results[resKey].name contains an empty list or does it contain 75 entries
                if len(results[resKey].name) == 0 or len(results[resKey].name[-1]) == 75:
                    # append a new empty list
                    results[resKey].name.append([])
                
                # append the name to the last list
                results[resKey].name[0].append(modelName)

            # Map accuracy here
            if "accuracy" in lineAcc and parseAccuracy == True:
                parseAccuracy = False
                line = line.split()
                accuracy = line[0]
                accuracy = round(float(accuracy), 5)

                
                # check if results[resKey].f1 contains an empty list or does it contain 3 entries
                if len(results[resKey].accuracy) == 0 or len(results[resKey].accuracy[-1]) == 3:
                    # append a new empty list
                    results[resKey].accuracy.append([])
                results[resKey].accuracy[-1].append(accuracy)
            

            # Check if line contains "Validation"
            if "Validation" in line:
                # remove "Validation" from the line
                line = line.replace("Validation", "")
                line = line.split(";")

                # NER f1
                f1Res = line[0].split(":")[1]

                # lr-f1
                logRegF1Res = line[1].split(":")[1]

                # loss
                lossRes = line[2].split(":")[1]

                # Extract string between "Language: " till end
                lng = line[3].split(":")[1].replace(" ", "")


                # convert f1 and loss to float and round it to fifth digit
                f1Res = round(float(f1Res), 5)
                logRegF1Res = round(float(logRegF1Res), 5)
                lossRes = round(float(lossRes), 5)

                
                # check if results[resKey].f1 contains an empty list or does it contain 75 entries
                if len(results[resKey].res) == 0 or len(results[resKey].res[-1]) == 3:
                    # append a new empty list
                    results[resKey].res.append([])
                
                # add f1 and loss to the last list in results[resKey].f1
                entry = (f1Res, logRegF1Res, lossRes, lng)
                results[resKey].res[-1].append(entry)
    
    plotResults(results, "Sequence Labeling")
    plotResults(results, "Binary Classification")






if __name__ == '__main__':
    main()