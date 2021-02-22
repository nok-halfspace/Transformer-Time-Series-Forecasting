import matplotlib.pyplot as plt
from helpers import EMA
from icecream import ic 
import numpy as np
import torch

def plot_loss(path_to_save, train=True):
    plt.rcParams.update({'font.size': 10})
    with open(path_to_save + "/train_loss.txt", 'r') as f:
        loss_list = [float(line) for line in f.readlines()]
    if train:
        title = "Train"
    else:
        title = "Validation"
    EMA_loss = EMA(loss_list)
    plt.plot(loss_list, label = "loss")
    plt.plot(EMA_loss, label="EMA loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(title+"_loss")
    plt.savefig(path_to_save+f"/{title}.png")
    plt.close()

def plot_prediction(title, path_to_save, src, tgt, prediction, sensor_number, index_in, index_tar):

    idx_scr = index_in[0, 1:].tolist()
    idx_tgt = index_tar[0].tolist()
    idx_pred = [i for i in range(idx_scr[0] +1, idx_tgt[-1])] #t2 - t61

    plt.figure(figsize=(15,6))
    plt.rcParams.update({"font.size" : 16})

    # connect with last elemenet in src
    # tgt = np.append(src[-1], tgt.flatten())
    # prediction = np.append(src[-1], prediction.flatten())

    # plotting
    plt.plot(idx_scr, src, '-', color = 'blue', label = 'Input', linewidth=2)
    plt.plot(idx_tgt, tgt, '-', color = 'indigo', label = 'Target', linewidth=2)
    plt.plot(idx_pred, prediction,'--', color = 'limegreen', label = 'Forecast', linewidth=2)

    #formatting
    plt.grid(b=True, which='major', linestyle = 'solid')
    plt.minorticks_on()
    plt.grid(b=True, which='minor', linestyle = 'dashed', alpha=0.5)
    plt.xlabel("Time Elapsed")
    plt.ylabel("Humidity (%)")
    plt.legend()
    plt.title("Forecast from Sensor " + str(sensor_number[0]))

    # save
    plt.savefig(path_to_save+f"Prediction_{title}.png")
    plt.close()

def plot_training(epoch, path_to_save, src, prediction, sensor_number, index_in, index_tar):

    # idx_scr = index_in.tolist()[0]
    # idx_tar = index_tar.tolist()[0]
    # idx_pred = idx_scr.append(idx_tar.append([idx_tar[-1] + 1]))

    idx_scr = [i for i in range(len(src))]
    idx_pred = [i for i in range(1, len(prediction)+1)]

    plt.figure(figsize=(15,6))
    plt.rcParams.update({"font.size" : 18})
    plt.grid(b=True, which='major', linestyle = '-')
    plt.grid(b=True, which='minor', linestyle = '--', alpha=0.5)
    plt.minorticks_on()

    plt.plot(idx_scr, src, 'o-.', color = 'blue', label = 'input sequence', linewidth=1)
    plt.plot(idx_pred, prediction, 'o-.', color = 'limegreen', label = 'prediction sequence', linewidth=1)

    plt.title("Teaching Forcing from Sensor " + str(sensor_number[0]) + ", Epoch " + str(epoch))
    plt.xlabel("Time Elapsed")
    plt.ylabel("Humidity (%)")
    plt.legend()
    plt.savefig(path_to_save+f"/Epoch_{str(epoch)}.png")
    plt.close()

def plot_training_3(epoch, path_to_save, src, sampled_src, prediction, sensor_number, index_in, index_tar):

    # idx_scr = index_in.tolist()[0]
    # idx_tar = index_tar.tolist()[0]
    # idx_pred = idx_scr.append(idx_tar.append([idx_tar[-1] + 1]))

    idx_scr = [i for i in range(len(src))]
    idx_pred = [i for i in range(1, len(prediction)+1)]
    idx_sampled_src = [i for i in range(len(sampled_src))]

    plt.figure(figsize=(15,6))
    plt.rcParams.update({"font.size" : 18})
    plt.grid(b=True, which='major', linestyle = '-')
    plt.grid(b=True, which='minor', linestyle = '--', alpha=0.5)
    plt.minorticks_on()

    ## REMOVE DROPOUT FOR THIS PLOT TO APPEAR AS EXPECTED !! DROPOUT INTERFERES WITH HOW THE SAMPLED SOURCES ARE PLOTTED
    plt.plot(idx_sampled_src, sampled_src, 'o-.', color='red', label = 'sampled source', linewidth=1, markersize=10)
    plt.plot(idx_scr, src, 'o-.', color = 'blue', label = 'input sequence', linewidth=1)
    plt.plot(idx_pred, prediction, 'o-.', color = 'limegreen', label = 'prediction sequence', linewidth=1)
    plt.title("Teaching Forcing from Sensor " + str(sensor_number[0]) + ", Epoch " + str(epoch))
    plt.xlabel("Time Elapsed")
    plt.ylabel("Humidity (%)")
    plt.legend()
    plt.savefig(path_to_save+f"/Epoch_{str(epoch)}.png")
    plt.close()