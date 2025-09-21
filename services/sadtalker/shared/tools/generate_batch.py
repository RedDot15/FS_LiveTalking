import os

from tqdm import tqdm
import torch
import numpy as np
import random
import scipy.io as scio
import shared.tools.features.audio as audio
from shared.logger import get_logger

logger = get_logger(__name__)

def crop_pad_audio(wav, audio_length):
    logger.debug(f"Cropping/padding audio: original length={len(wav)}, target length={audio_length}")
    if len(wav) > audio_length:
        wav = wav[:audio_length]
        logger.debug("Audio cropped to target length")
    elif len(wav) < audio_length:
        wav = np.pad(wav, [0, audio_length - len(wav)], mode='constant', constant_values=0)
        logger.debug("Audio padded to target length")
    return wav

def parse_audio_length(audio_length, sr, fps):
    logger.debug(f"Parsing audio length: audio_length={audio_length}, sr={sr}, fps={fps}")
    bit_per_frames = sr / fps

    num_frames = int(audio_length / bit_per_frames)
    audio_length = int(num_frames * bit_per_frames)
    
    logger.debug(f"Parsed audio: num_frames={num_frames}, adjusted_audio_length={audio_length}")
    return audio_length, num_frames

def generate_blink_seq(num_frames):
    logger.debug(f"Generating regular blink sequence for {num_frames} frames")
    ratio = np.zeros((num_frames,1))
    frame_id = 0
    blink_count = 0
    while frame_id in range(num_frames):
        start = 80
        if frame_id+start+9<=num_frames - 1:
            ratio[frame_id+start:frame_id+start+9, 0] = [0.5,0.6,0.7,0.9,1, 0.9, 0.7,0.6,0.5]
            frame_id = frame_id+start+9
            blink_count += 1
        else:
            break
    logger.debug(f"Generated {blink_count} blink sequences")
    return ratio 

def generate_blink_seq_randomly(num_frames):
    logger.debug(f"Generating random blink sequence for {num_frames} frames")
    ratio = np.zeros((num_frames,1))
    if num_frames<=20:
        logger.debug("Too few frames for blinking, returning zero sequence")
        return ratio
    frame_id = 0
    blink_count = 0
    while frame_id in range(num_frames):
        start = random.choice(range(min(10,num_frames), min(int(num_frames/2), 70))) 
        if frame_id+start+5<=num_frames - 1:
            ratio[frame_id+start:frame_id+start+5, 0] = [0.5, 0.9, 1.0, 0.9, 0.5]
            frame_id = frame_id+start+5
            blink_count += 1
        else:
            break
    logger.debug(f"Generated {blink_count} random blink sequences")
    return ratio

def get_data(first_coeff_path, audio_path, device, ref_eyeblink_coeff_path, still=False, idlemode=False, length_of_audio=False, use_blink=True):
    logger.info(f"Getting data for batch processing: first_coeff_path='{first_coeff_path}', audio_path='{audio_path}', device='{device}'")
    logger.debug(f"Parameters: still={still}, idlemode={idlemode}, length_of_audio={length_of_audio}, use_blink={use_blink}")

    syncnet_mel_step_size = 16
    fps = 25

    pic_name = os.path.splitext(os.path.split(first_coeff_path)[-1])[0]
    audio_name = os.path.splitext(os.path.split(audio_path)[-1])[0]
    logger.debug(f"Processing: pic_name='{pic_name}', audio_name='{audio_name}'")

    
    if idlemode:
        logger.info(f"Using idle mode: generating {length_of_audio * 25} frames")
        num_frames = int(length_of_audio * 25)
        indiv_mels = np.zeros((num_frames, 80, 16))
    else:
        logger.info(f"Loading and processing audio: {audio_path}")
        wav = audio.load_wav(audio_path, 16000) 
        wav_length, num_frames = parse_audio_length(len(wav), 16000, 25)
        wav = crop_pad_audio(wav, wav_length)
        logger.info(f"Audio processed: {num_frames} frames, wav_length={wav_length}")
        
        orig_mel = audio.melspectrogram(wav).T
        spec = orig_mel.copy()         # nframes 80
        indiv_mels = []

        logger.info("Generating mel spectrograms for each frame")
        for i in tqdm(range(num_frames), 'mel:'):
            start_frame_num = i-2
            start_idx = int(80. * (start_frame_num / float(fps)))
            end_idx = start_idx + syncnet_mel_step_size
            seq = list(range(start_idx, end_idx))
            seq = [ min(max(item, 0), orig_mel.shape[0]-1) for item in seq ]
            m = spec[seq, :]
            indiv_mels.append(m.T)
        indiv_mels = np.asarray(indiv_mels)         # T 80 16
        logger.info("Mel spectrograms generated successfully")

    ratio = generate_blink_seq_randomly(num_frames)      # T
    source_semantics_path = first_coeff_path
    logger.debug(f"Loading source semantics from: {source_semantics_path}")
    source_semantics_dict = scio.loadmat(source_semantics_path)
    ref_coeff = source_semantics_dict['coeff_3dmm'][:1,:70]         #1 70
    ref_coeff = np.repeat(ref_coeff, num_frames, axis=0)
    logger.debug(f"Source semantics loaded and repeated for {num_frames} frames")

    if ref_eyeblink_coeff_path is not None:
        logger.info(f"Using reference eyeblink coefficients from: {ref_eyeblink_coeff_path}")
        ratio[:num_frames] = 0
        refeyeblink_coeff_dict = scio.loadmat(ref_eyeblink_coeff_path)
        refeyeblink_coeff = refeyeblink_coeff_dict['coeff_3dmm'][:,:64]
        refeyeblink_num_frames = refeyeblink_coeff.shape[0]
        logger.debug(f"Reference eyeblink frames: {refeyeblink_num_frames}, target frames: {num_frames}")
        
        if refeyeblink_num_frames<num_frames:
            logger.debug("Extending reference eyeblink coefficients to match target length")
            div = num_frames//refeyeblink_num_frames
            re = num_frames%refeyeblink_num_frames
            refeyeblink_coeff_list = [refeyeblink_coeff for i in range(div)]
            refeyeblink_coeff_list.append(refeyeblink_coeff[:re, :64])
            refeyeblink_coeff = np.concatenate(refeyeblink_coeff_list, axis=0)
            print(refeyeblink_coeff.shape[0])
            logger.debug(f"Extended eyeblink coefficients to {refeyeblink_coeff.shape[0]} frames")

        ref_coeff[:, :64] = refeyeblink_coeff[:num_frames, :64] 
        logger.info("Reference eyeblink coefficients applied successfully")
    
    logger.debug("Converting arrays to tensors and moving to device")
    indiv_mels = torch.FloatTensor(indiv_mels).unsqueeze(1).unsqueeze(0) # bs T 1 80 16

    if use_blink:
        ratio = torch.FloatTensor(ratio).unsqueeze(0)                       # bs T
        logger.debug("Using blink ratios")
    else:
        ratio = torch.FloatTensor(ratio).unsqueeze(0).fill_(0.) 
        logger.debug("Disabled blinking (filled with zeros)")
                               # bs T
    ref_coeff = torch.FloatTensor(ref_coeff).unsqueeze(0)                # bs 1 70

    indiv_mels = indiv_mels.to(device)
    ratio = ratio.to(device)
    ref_coeff = ref_coeff.to(device)
    logger.info(f"Data preparation completed successfully. Tensors moved to device: {device}")

    return {'indiv_mels': indiv_mels,  
            'ref': ref_coeff, 
            'num_frames': num_frames, 
            'ratio_gt': ratio,
            'audio_name': audio_name, 'pic_name': pic_name}

