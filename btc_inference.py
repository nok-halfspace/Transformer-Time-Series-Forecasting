from model import Transformer
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import logging
import time # debugging
from plot import plot_btc_prediction
# from helpers import *
from joblib import load
from icecream import ic
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s %(message)s", datefmt="[%Y-%m-%d %H:%M:%S]")
logger = logging.getLogger(__name__)

def log_loss(path_to_save, loss):
    with open(path_to_save + "/validation_loss.txt", 'a') as theFile:
        theFile.write(loss)
        theFile.write('\n')


def inference(path_to_save_predictions, forecast_window, dataloader, device, path_to_save_model, best_model):

    device = torch.device(device)
    
    model = Transformer().double().to(device)
    model.load_state_dict(torch.load(path_to_save_model+best_model))
    criterion = torch.nn.MSELoss()

    val_loss = 0
    with torch.no_grad(): #Context-manager that disabled gradient calculation.
        # the common practice for evaluating/validation is using torch.no_grad() in pair with model.eval() to turn off gradients computation
        model.eval()
        for plot in range(25):
#         with tqdm(total=len(dataloader.dataset), desc=f"[Epoch {epoch+1:3d}/{EPOCH}]") as pbar:
            dataset_len = len(dataloader.dataset)
            logger.info('data len is {}'.format(dataset_len))
            with tqdm(total=dataset_len, desc=f"[PLOT {plot}/{25}]") as pbar:
                for idx, (index_in, index_tar, _input, target) in enumerate(dataloader):
                # starting from 1 so that src matches with target, but has same length as when training
                    
                    pbar.set_postfix({'data length': (idx+1)})
                    pbar.update(index_in.shape[0])

                    src = _input.permute(1,0,2).double().to(device)[1:, :, :] # 47, 1, 7: t1 -- t47
                    target = target.permute(1,0,2).double().to(device) # t48 - t59

                    next_input_model = src
                    all_predictions = []

                    for i in range(forecast_window - 1):

                        prediction = model(next_input_model, device) # 47,1,1: t2' - t48'

                        if all_predictions == []:
                            all_predictions = prediction # 47,1,1: t2' - t48'
                        else:
                            all_predictions = torch.cat((all_predictions, prediction[-1,:,:].unsqueeze(0))) # 47+,1,1: t2' - t48', t49', t50'

                        pos_encoding_old_vals = src[i+1:, :, 1:] # 46, 1, 6, pop positional encoding first value: t2 -- t47
                        pos_encoding_new_val = target[i + 1, :, 1:].unsqueeze(1) # 1, 1, 6, append positional encoding of last predicted value: t48
                        pos_encodings = torch.cat((pos_encoding_old_vals, pos_encoding_new_val)) # 47, 1, 6 positional encodings matched with prediction: t2 -- t48

                        next_input_model = torch.cat((src[i+1:, :, 0].unsqueeze(-1), prediction[-1,:,:].unsqueeze(0))) #t2 -- t47, t48'
                        next_input_model = torch.cat((next_input_model, pos_encodings), dim = 2) # 47, 1, 7 input for next round

        #             logger.info('prediction length is {}'.format(len(all_predictions))) 69
                    true = torch.cat((src[1:,:,0],target[:-1,:,0]))
                    loss = criterion(true, all_predictions[:,:,0])
                    log_loss("save_loss/", str(loss.item()))
                    val_loss += loss
            
                val_loss = val_loss/10
                scaler = load('scalar_item.joblib')
                src_close_price = scaler.inverse_transform(src[:,:,0].cpu())
                target_close_price = scaler.inverse_transform(target[:,:,0].cpu())
                prediction_close_price = scaler.inverse_transform(all_predictions[:,:,0].detach().cpu().numpy())
        #         plot_btc_prediction(plot, path_to_save_predictions, src_close_price, target_close_price, prediction_close_price, index_in, index_tar)
                plot_btc_prediction(plot, path_to_save_predictions, src_close_price, target_close_price, prediction_close_price, index_in, index_tar)
            
        logger.info(f"Loss On Unseen Dataset: {val_loss.item()}")