from tqdm import tqdm
import torch
from torch import nn
from shared.logger import get_logger

logger = get_logger(__name__)


class Audio2Exp(nn.Module):
    def __init__(self, netG, cfg, device, prepare_training_loss=False):
        super(Audio2Exp, self).__init__()
        logger.info(f"Initializing Audio2Exp on device: {device}")
        logger.debug(f"Config: {cfg}")
        self.cfg = cfg
        self.device = device
        self.netG = netG.to(device)
        logger.info("Audio2Exp initialization completed")

    def test(self, batch):
        logger.info("Starting Audio2Exp inference")
        mel_input = batch['indiv_mels']                         # bs T 1 80 16
        bs = mel_input.shape[0]
        T = mel_input.shape[1]
        logger.debug(f"Input mel shape: {mel_input.shape}, batch_size: {bs}, time_steps: {T}")

        exp_coeff_pred = []

        logger.info(f"Processing {T} frames in chunks of 10")
        for i in tqdm(range(0, T, 10),'audio2exp:'): # every 10 frames
            
            current_mel_input = mel_input[:,i:i+10]

            #ref = batch['ref'][:, :, :64].repeat((1,current_mel_input.shape[1],1))           #bs T 64
            ref = batch['ref'][:, :, :64][:, i:i+10]
            ratio = batch['ratio_gt'][:, i:i+10]                               #bs T

            audiox = current_mel_input.view(-1, 1, 80, 16)                  # bs*T 1 80 16

            curr_exp_coeff_pred  = self.netG(audiox, ref, ratio)         # bs T 64 

            exp_coeff_pred += [curr_exp_coeff_pred]

        logger.debug(f"Generated {len(exp_coeff_pred)} expression coefficient chunks")
        # BS x T x 64
        final_exp_coeff = torch.cat(exp_coeff_pred, axis=1)
        logger.info(f"Audio2Exp inference completed. Final shape: {final_exp_coeff.shape}")
        
        results_dict = {
            'exp_coeff_pred': final_exp_coeff
            }
        return results_dict


