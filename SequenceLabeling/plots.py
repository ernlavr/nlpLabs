from typing import List
import matplotlib.pyplot as plt
import os
from dataclasses import dataclass



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
filePath = os.path.join(dir_path, 'lab5_results.txt')



@dataclass
class DataPoint():
    name : list
    classWeights : list
    cwDivisor : list
    f1 : list
    loss : list
    epoch : list

def plotResults(datapoint : dict):
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
    for model in range(0, 6):            
        # create an empty lineplot canvas, x axis should be epochs 1 to 75, there should be two y axis for f1 and loss
        fig, ax = plt.subplots()
        
        # create a plot for each language
        for language in datapoint:
            # get the name of the model
            modelName = datapoint[language].name[model][0]
            # get the class weights and divisor
            classWeights = datapoint[language].classWeights[model]
            cwDivisor = datapoint[language].cwDivisor[model]
            # get the f1 and loss
            f1 = datapoint[language].f1[model]
            loss = datapoint[language].loss[model]

            # plot the f1 and loss on two different y axis and denote epochs on x axis
            # label should denote 
            ax.plot(f1, label=f"F1_{language}") 
            # highlight hte maximum F1 value and the epoch it occured
            maxF1 = max(f1)
            maxF1Epoch = f1.index(maxF1)
            # Print the best epoch and the corresponding language and model
            print(f"Best F1 for {modelName} with {language} is {maxF1} at epoch {maxF1Epoch}")

            ax.scatter(maxF1Epoch, maxF1, label=f"Max F1_{language}")

            ax.plot(loss, label=f"Loss_{language}")
            # highlight the minimum loss value and the epoch it occured
            minLoss = min(loss)
            minLossEpoch = loss.index(minLoss)
            ax.scatter(minLossEpoch, minLoss, label=f"Min Loss_{language}")

            # write down the maxF1 score in a plot description
            ax.text(maxF1Epoch, maxF1, f"{maxF1:.2f}")
            
            # write down the minLoss score in a plot description
            ax.text(minLossEpoch, minLoss, f"{minLoss:.2f}")

        # add a legend to the plot
        ax.legend()

        # fix the Y axis to be between 0 to 1 if modelName include "basic", else 0 to 0.5
        if "basic" in modelName:
            ax.set_ylim(0, 1)
        else:
            ax.set_ylim(0, 0.33)
        
        # split the modelName by "_"
        modelName = modelName.split("_")[1]

        # set the title to denote class weights and divisor
        ax.set_title(f"{modelName}")
        
        # zoom out the plot and place the legend outside the plot
        fig.set_size_inches(5.1, 4.25)
        # place the legend outside the plot
        ax.legend(bbox_to_anchor=(1.04, 1), loc='upper left', borderaxespad=0.)

        # annotate x axis label as epoch
        ax.set_xlabel("Epoch")
        # annotate y axis label as f1 and loss
        ax.set_ylabel("F1 and Loss")
        # shift the plot off-center
        ax.set_position([0.1, 0.1, 0.55, 0.85])

        # always show the last number on x axis 75
        ax.set_xticks(range(0, 75, 10))

        
        # save the plot to a file if cwDivisor is (1, 1)
        if cwDivisor == (1, 1):
            plt.savefig(f"{modelName}.png")









    


def main(): 
    language = ""
    currModel = ""
    results = {}

    results["english"] = DataPoint([], [], [], [], [], [])
    results["finnish"] = DataPoint([], [], [], [], [], [])
    results["japanese"] = DataPoint([], [], [], [], [], [])

    # read and parse file called log
    with open(filePath) as f:
        # parse each line on a loop
        # create a line plot canvas

        for line in f:
            # remove timestamp and newline
            line = line[17:-1]

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
                results[language].cwDivisor.append(cwDivisor)
                results[language].cwDivisor.append(cwDivisor)

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
                results[language].classWeights.append(classWeights)
                results[language].classWeights.append(classWeights) 


            # Check if line contains "Model"
            if "Model" in line:
                # extract the substring between : and ;
                modelName = line.split(":")[1].split(";")[0]
                
                # check if results[language].name contains an empty list or does it contain 75 entries
                if len(results[language].name) == 0 or len(results[language].name[-1]) == 75:
                    # append a new empty list
                    results[language].name.append([])
                
                # append the name to the last list
                results[language].name[-1].append(modelName)

            # Check if line contains "Validation"
            if "Validation" in line:
                # extract numerical string between "F1:" and ","
                f1Res = line[line.find("F1:")+3:line.find(",")]
                # extract numerical string between "train loss:" and till end
                lossRes = line[line.find("train loss:")+11:]
                # convert f1 and loss to float and round it to fifth digit
                f1Res = round(float(f1Res), 5)
                lossRes = round(float(lossRes), 5)

                
                # check if results[language].f1 contains an empty list or does it contain 75 entries
                if len(results[language].f1) == 0 or len(results[language].f1[-1]) == 75:
                    # append a new empty list
                    results[language].f1.append([])
                    results[language].loss.append([])
                
                # add f1 and loss to the last list in results[language].f1
                results[language].f1[-1].append(f1Res)
                results[language].loss[-1].append(lossRes)            
    
    plotResults(results)






if __name__ == '__main__':
    main()