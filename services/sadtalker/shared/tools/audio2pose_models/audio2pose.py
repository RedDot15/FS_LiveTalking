import torch
from torch import nn

from shared.tools.audio2pose_models.cvae import CVAE
from shared.tools.audio2pose_models.discriminator import PoseSequenceDiscriminator
from shared.tools.audio2pose_models.audio_encoder import AudioEncoder
from shared.logger import get_logger

logger = get_logger(__name__)

class Audio2Pose(nn.Module):
    def __init__(self, cfg, wav2lip_checkpoint, device='cuda'):
        super().__init__()
        logger.info(f"Initializing Audio2Pose on device: {device}")
        self.cfg = cfg
        self.seq_len = cfg.MODEL.CVAE.SEQ_LEN
        self.latent_dim = cfg.MODEL.CVAE.LATENT_SIZE
        self.device = device
        logger.debug(f"Sequence length: {self.seq_len}, Latent dimension: {self.latent_dim}")

        logger.info("Loading audio encoder")
        self.audio_encoder = AudioEncoder(wav2lip_checkpoint, device)
        self.audio_encoder.eval()
        for param in self.audio_encoder.parameters():
            param.requires_grad = False
        logger.debug("Audio encoder loaded and set to eval mode")

        logger.info("Initializing CVAE and discriminator")
        self.netG = CVAE(cfg)
        self.netD_motion = PoseSequenceDiscriminator(cfg)
        logger.info("Audio2Pose initialization completed")
        
        
    def forward(self, x):

        batch = {}
        coeff_gt = x['gt'].cuda().squeeze(0)           #bs frame_len+1 73
        batch['pose_motion_gt'] = coeff_gt[:, 1:, 64:70] - coeff_gt[:, :1, 64:70] #bs frame_len 6
        batch['ref'] = coeff_gt[:, 0, 64:70]  #bs  6
        batch['class'] = x['class'].squeeze(0).cuda() # bs
        indiv_mels= x['indiv_mels'].cuda().squeeze(0) # bs seq_len+1 80 16

        # forward
        audio_emb_list = []
        audio_emb = self.audio_encoder(indiv_mels[:, 1:, :, :].unsqueeze(2)) #bs seq_len 512
        batch['audio_emb'] = audio_emb
        batch = self.netG(batch)

        pose_motion_pred = batch['pose_motion_pred']           # bs frame_len 6
        pose_gt = coeff_gt[:, 1:, 64:70].clone()               # bs frame_len 6
        pose_pred = coeff_gt[:, :1, 64:70] + pose_motion_pred  # bs frame_len 6

        batch['pose_pred'] = pose_pred
        batch['pose_gt'] = pose_gt

        return batch

    def test(self, x):
        logger.info("Starting Audio2Pose test inference")
        batch = {}
        ref = x['ref']                            #bs 1 70
        batch['ref'] = x['ref'][:,0,-6:]  
        batch['class'] = x['class']  
        bs = ref.shape[0]
        logger.debug(f"Batch size: {bs}, Reference shape: {ref.shape}")
        
        indiv_mels= x['indiv_mels']               # bs T 1 80 16
        indiv_mels_use = indiv_mels[:, 1:]        # we regard the ref as the first frame
        num_frames = x['num_frames']
        num_frames = int(num_frames) - 1
        logger.debug(f"Processing {num_frames} frames, mel input shape: {indiv_mels.shape}")

        #  
        div = num_frames//self.seq_len
        re = num_frames%self.seq_len
        logger.debug(f"Sequence division: {div} full sequences, {re} remaining frames")
        
        audio_emb_list = []
        pose_motion_pred_list = [torch.zeros(batch['ref'].unsqueeze(1).shape, dtype=batch['ref'].dtype, 
                                                device=batch['ref'].device)]

        logger.info(f"Processing {div} full sequences")
        for i in range(div):
            z = torch.randn(bs, self.latent_dim).to(ref.device)
            batch['z'] = z
            audio_emb = self.audio_encoder(indiv_mels_use[:, i*self.seq_len:(i+1)*self.seq_len,:,:,:]) #bs seq_len 512
            batch['audio_emb'] = audio_emb
            batch = self.netG.test(batch)
            pose_motion_pred_list.append(batch['pose_motion_pred'])  #list of bs seq_len 6
        
        if re != 0:
            logger.debug(f"Processing remaining {re} frames")
            z = torch.randn(bs, self.latent_dim).to(ref.device)
            batch['z'] = z
            audio_emb = self.audio_encoder(indiv_mels_use[:, -1*self.seq_len:,:,:,:]) #bs seq_len  512
            if audio_emb.shape[1] != self.seq_len:
                pad_dim = self.seq_len-audio_emb.shape[1]
                pad_audio_emb = audio_emb[:, :1].repeat(1, pad_dim, 1) 
                audio_emb = torch.cat([pad_audio_emb, audio_emb], 1) 
                logger.debug(f"Padded audio embedding from {audio_emb.shape[1] - pad_dim} to {audio_emb.shape[1]} frames")
            batch['audio_emb'] = audio_emb
            batch = self.netG.test(batch)
            pose_motion_pred_list.append(batch['pose_motion_pred'][:,-1*re:,:])   
        
        pose_motion_pred = torch.cat(pose_motion_pred_list, dim = 1)
        batch['pose_motion_pred'] = pose_motion_pred

        pose_pred = ref[:, :1, -6:] + pose_motion_pred  # bs T 6

        batch['pose_pred'] = pose_pred
        logger.info(f"Audio2Pose test completed. Pose prediction shape: {pose_pred.shape}")
        return batch
