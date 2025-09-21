import os 
import torch
import numpy as np
from scipy.io import savemat, loadmat
from yacs.config import CfgNode as CN
from scipy.signal import savgol_filter

import safetensors
import safetensors.torch 

from shared.tools.audio2pose_models.audio2pose import Audio2Pose
from shared.tools.audio2exp_models.networks import SimpleWrapperV2
from shared.tools.audio2exp_models.audio2exp import Audio2Exp
from shared.tools.features.safetensor_helper import load_x_from_safetensor
from shared.logger import get_logger

logger = get_logger(__name__)


def load_cpk(checkpoint_path, model=None, optimizer=None, device="cpu"):
    logger.debug(f"Loading checkpoint from: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
        if model is not None:
            model.load_state_dict(checkpoint['model'])
            logger.debug("Model state loaded successfully")
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.debug("Optimizer state loaded successfully")
        
        epoch = checkpoint['epoch']
        logger.info(f"Checkpoint loaded successfully from epoch: {epoch}")
        return epoch
    except Exception as e:
        logger.error(f"Failed to load checkpoint from {checkpoint_path}: {e}")
        raise

class Audio2Coeff():

    def __init__(self, sadtalker_path, device):
        logger.info(f"Initializing Audio2Coeff on device: {device}")
        logger.debug(f"SadTalker paths: {sadtalker_path}")
        
        #load config
        logger.info("Loading audio2pose and audio2exp configurations")
        fcfg_pose = open(sadtalker_path['audio2pose_yaml_path'])
        cfg_pose = CN.load_cfg(fcfg_pose)
        cfg_pose.freeze()
        fcfg_exp = open(sadtalker_path['audio2exp_yaml_path'])
        cfg_exp = CN.load_cfg(fcfg_exp)
        cfg_exp.freeze()
        logger.debug("Configurations loaded and frozen")

        # load audio2pose_model
        logger.info("Loading audio2pose model")
        self.audio2pose_model = Audio2Pose(cfg_pose, None, device=device)
        self.audio2pose_model = self.audio2pose_model.to(device)
        self.audio2pose_model.eval()
        for param in self.audio2pose_model.parameters():
            param.requires_grad = False 
        logger.debug("Audio2pose model loaded and set to eval mode")
        
        try:
            if sadtalker_path['use_safetensor']:
                logger.info("Loading audio2pose model from safetensor")
                checkpoints = safetensors.torch.load_file(sadtalker_path['checkpoint'])
                self.audio2pose_model.load_state_dict(load_x_from_safetensor(checkpoints, 'audio2pose'))
            else:
                logger.info("Loading audio2pose model from checkpoint")
                load_cpk(sadtalker_path['audio2pose_checkpoint'], model=self.audio2pose_model, device=device)
            logger.info("Audio2pose model weights loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load audio2pose checkpoint: {e}")
            raise Exception("Failed in loading audio2pose_checkpoint")

        # load audio2exp_model
        logger.info("Loading audio2exp model")
        netG = SimpleWrapperV2()
        netG = netG.to(device)
        for param in netG.parameters():
            netG.requires_grad = False
        netG.eval()
        logger.debug("Audio2exp netG model loaded and set to eval mode")
        
        try:
            if sadtalker_path['use_safetensor']:
                logger.info("Loading audio2exp model from safetensor")
                checkpoints = safetensors.torch.load_file(sadtalker_path['checkpoint'])
                netG.load_state_dict(load_x_from_safetensor(checkpoints, 'audio2exp'))
            else:
                logger.info("Loading audio2exp model from checkpoint")
                load_cpk(sadtalker_path['audio2exp_checkpoint'], model=netG, device=device)
            logger.info("Audio2exp model weights loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load audio2exp checkpoint: {e}")
            raise Exception("Failed in loading audio2exp_checkpoint")
            
        self.audio2exp_model = Audio2Exp(netG, cfg_exp, device=device, prepare_training_loss=False)
        self.audio2exp_model = self.audio2exp_model.to(device)
        for param in self.audio2exp_model.parameters():
            param.requires_grad = False
        self.audio2exp_model.eval()
        logger.debug("Audio2exp model wrapper initialized and set to eval mode")
 
        self.device = device
        logger.info("Audio2Coeff initialization completed successfully")

    def generate(self, batch, coeff_save_dir, pose_style, ref_pose_coeff_path=None):
        logger.info(f"Generating coefficients with pose_style={pose_style}, ref_pose_coeff_path='{ref_pose_coeff_path}'")
        logger.debug(f"Batch keys: {batch.keys()}, save_dir: {coeff_save_dir}")

        with torch.no_grad():
            #test
            logger.info("Running audio2exp model inference")
            results_dict_exp= self.audio2exp_model.test(batch)
            exp_pred = results_dict_exp['exp_coeff_pred']                         #bs T 64
            logger.debug(f"Expression prediction shape: {exp_pred.shape}")

            #for class_id in  range(1):
            #class_id = 0#(i+10)%45
            #class_id = random.randint(0,46)                                   #46 styles can be selected 
            batch['class'] = torch.LongTensor([pose_style]).to(self.device)
            logger.info("Running audio2pose model inference")
            results_dict_pose = self.audio2pose_model.test(batch) 
            pose_pred = results_dict_pose['pose_pred']                        #bs T 6
            logger.debug(f"Pose prediction shape: {pose_pred.shape}")

            pose_len = pose_pred.shape[1]
            logger.debug(f"Applying Savitzky-Golay filter to pose predictions, length: {pose_len}")
            if pose_len<13: 
                pose_len = int((pose_len-1)/2)*2+1
                pose_pred = torch.Tensor(savgol_filter(np.array(pose_pred.cpu()), pose_len, 2, axis=1)).to(self.device)
                logger.debug(f"Applied SG filter with window length: {pose_len}")
            else:
                pose_pred = torch.Tensor(savgol_filter(np.array(pose_pred.cpu()), 13, 2, axis=1)).to(self.device) 
                logger.debug("Applied SG filter with window length: 13")
            
            coeffs_pred = torch.cat((exp_pred, pose_pred), dim=-1)            #bs T 70
            logger.debug(f"Combined coefficients shape: {coeffs_pred.shape}")

            coeffs_pred_numpy = coeffs_pred[0].clone().detach().cpu().numpy() 

            if ref_pose_coeff_path is not None: 
                logger.info(f"Applying reference pose from: {ref_pose_coeff_path}")
                coeffs_pred_numpy = self.using_refpose(coeffs_pred_numpy, ref_pose_coeff_path)
        
            output_path = os.path.join(coeff_save_dir, '%s##%s.mat'%(batch['pic_name'], batch['audio_name']))
            logger.info(f"Saving coefficients to: {output_path}")
            savemat(output_path, {'coeff_3dmm': coeffs_pred_numpy})
            logger.info("Coefficients generation completed successfully")

            return output_path
    
    def using_refpose(self, coeffs_pred_numpy, ref_pose_coeff_path):
        logger.info(f"Applying reference pose from: {ref_pose_coeff_path}")
        num_frames = coeffs_pred_numpy.shape[0]
        logger.debug(f"Target frames: {num_frames}")
        
        refpose_coeff_dict = loadmat(ref_pose_coeff_path)
        refpose_coeff = refpose_coeff_dict['coeff_3dmm'][:,64:70]
        refpose_num_frames = refpose_coeff.shape[0]
        logger.debug(f"Reference pose frames: {refpose_num_frames}")
        
        if refpose_num_frames<num_frames:
            logger.debug("Extending reference pose to match target length")
            div = num_frames//refpose_num_frames
            re = num_frames%refpose_num_frames
            refpose_coeff_list = [refpose_coeff for i in range(div)]
            refpose_coeff_list.append(refpose_coeff[:re, :])
            refpose_coeff = np.concatenate(refpose_coeff_list, axis=0)
            logger.debug(f"Extended reference pose to {refpose_coeff.shape[0]} frames")

        #### relative head pose
        logger.debug("Applying relative head pose transformation")
        coeffs_pred_numpy[:, 64:70] = coeffs_pred_numpy[:, 64:70] + ( refpose_coeff[:num_frames, :] - refpose_coeff[0:1, :] )
        logger.info("Reference pose applied successfully")
        return coeffs_pred_numpy


