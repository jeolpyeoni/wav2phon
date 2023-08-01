import torch
import argparse
import math
import copy
import csv
import numpy as np
import soundfile as sf
import ffmpeg

from pydub import AudioSegment
from tqdm import tqdm

from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from datasets import load_dataset


import warnings
warnings.filterwarnings(action='ignore')



def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    
    '''
    <Input>
    t:            index / step            (float)
    initial_lr:                           (float)
    '''

    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)


    return initial_lr * lr_ramp



def optim_model(args, processor, model, input_audio, sample_rate, target_transcription):
    
    ## Set parameters
    initial_lr = 8e-6
    threshold = 0.1
    
    
    ## Model initialization
    input_values = processor(input_audio, sampling_rate=sample_rate, return_tensors="pt").input_values.cuda()
    
    with processor.as_target_processor():
        labels = processor(target_transcription, return_tensors="pt").input_ids
    
    model.freeze_feature_encoder()
    optim = torch.optim.AdamW(
        params = model.parameters(),
        lr = initial_lr
    )

    
    ## Optimize
    pbar = tqdm(range(args.steps))
        
    for i in pbar:

        t = i / args.steps
        lr = get_lr(t, initial_lr)

        optim.param_groups[0]["lr"] = lr

        
        ## Update loss
        loss = model(input_values, labels=labels).loss
        optim.zero_grad()
        loss.backward()
        optim.step()
        
        
        ## Print loss
        pbar.set_description(
            f"loss: {loss.item():.4f}"
        )
        
        if loss.item() <= threshold:
            break
            
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    
    
    return predicted_ids
    

    
def find_word(args, processor, input_audio, sample_rate, pred, target_transcription, start_idx, end_idx):
    

    ## List predicted labels
    pred_labels = []
    for i in range(pred.size(1)):
        temp = pred[:,i]
        pred_labels.append(int(temp))
        
    
    ## List nonzero labels and indices
    nonzero_labels = []
    nonzero_indices = []
    for i in range(len(pred_labels)):
        if pred_labels[i] != 0:
            nonzero_labels.append(pred_labels[i])
            nonzero_indices.append(i)
        

    ## Parameters initialization
    win_interv = (len(input_audio) / pred.size(1)) / sample_rate
    words = target_transcription.split(" ")
    
    word_info = []
    start_info = []
    end_info = []
    interv_info = []
    duration_info = []
    
    last_index = -1

    
    ## Build word-phoneme dictionary
    words_with_phonemes = {}
    for j in range(len(words)):
        
        word = words[j]
        with processor.as_target_processor():
            labels = processor(word, return_tensors="pt").input_ids
        labels = labels.tolist()[0]
        
        if word not in words_with_phonemes:
            words_with_phonemes[word] = labels
            

    ## Find fully matching sequences
    word_indices = []
    word_search_index = 0
    found_indices = []
    
    for i in range(len(words)):
            
        found = False
        
        word = words[i]
        gt_phonemes = words_with_phonemes[word]
        
        
        if (word_search_index+len(gt_phonemes) <= len(nonzero_labels)):
            if (all(nonzero_labels[word_search_index+j] == phoneme for j, phoneme in enumerate(gt_phonemes))):

                start = word_search_index
                end = start + len(gt_phonemes) - 1

                found = True
                found_indices.append(i)
                word_indices.append([word, start, end])
                word_search_index = end + 1
            
        
        if (found == False) and (word_search_index+1+len(gt_phonemes) <= len(nonzero_labels)):
            if (all(nonzero_labels[word_search_index+1+j] == phoneme for j, phoneme in enumerate(gt_phonemes))):

                start = word_search_index + 1
                end = start + len(gt_phonemes) - 1

                found = True
                found_indices.append(i)
                word_indices.append([word, start, end])
                word_search_index = end + 1
                
        
        if (found == False) and (word_search_index+2+len(gt_phonemes) <= len(nonzero_labels)):
            if (all(nonzero_labels[word_search_index+2+j] == phoneme for j, phoneme in enumerate(gt_phonemes))):

                start = word_search_index + 2
                end = start + len(gt_phonemes) - 1

                found = True
                found_indices.append(i)
                word_indices.append([word, start, end])
                word_search_index = end + 1
            
            
        if (found == False):
            word_search_index += len(gt_phonemes)

    
    ## Find similar sequences for unfound words
    word_count = 0
    for word in words:
    
        found = False
        gt_phonemes = words_with_phonemes[word]
        
        
        if word_count not in found_indices:
            
            if word_count != 0:
                prev_end = word_indices[word_count - 1][2]
                start = prev_end + 1

                gt_phonemes = words_with_phonemes[word]
                
                
                if len(gt_phonemes) == 1 or ((nonzero_labels[start] == gt_phonemes[0]) or (nonzero_labels[start+len(gt_phonemes)-1] == gt_phonemes[-1])):
                    pass
                
                else:
                    if ((nonzero_labels[start+1] == gt_phonemes[0]) or (nonzero_labels[start+len(gt_phonemes)] == gt_phonemes[-1])):
                        start += 1
                        found = True
                        
                    if (found == False) and ((nonzero_labels[start+2] == gt_phonemes[0]) or (nonzero_labels[start+len(gt_phonemes)+1] == gt_phonemes[-1])):
                        start += 2
                        found = True
        
                end = start + len(gt_phonemes) - 1
                
                
            else:
                next_start = next_start = word_indices[word_count][1]
                end = next_start - 1
                start = end - (len(gt_phonemes) - 1)
            
            word_indices.insert(word_count, [word, start, end])
            
        word_count += 1
        
    
    ## Save timing infos
    for i in range(len(word_indices)):
        
        word, start, end = word_indices[i]
        start_timing = start_idx + (nonzero_indices[start] * win_interv)
        end_timing = start_idx + (nonzero_indices[end] * win_interv)
        
        word_info.append((f"{word}", start_timing, end_timing))
        start_info.append(start_timing)
        end_info.append(end_timing)
        duration_info.append(end_timing - start_timing)

    for i in range(1, len(word_info)):
        prev_end = word_info[i-1][2]
        cur_start = word_info[i][1]
        interv_info.append(float(cur_start) - float(prev_end))
        
                                    
    return start_info, end_info, duration_info, interv_info                
            
                        


if __name__ == "__main__":
    
    
    ## Set arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--steps', type=int, default=50)
    parser.add_argument('-a', '--audio_dir', type=str, default='')
    parser.add_argument('-t', '--text_dir', type=str, default='')
    parser.add_argument('-fr', '--frame_rate', type=int, default=60)
    args = parser.parse_args()
    
    
    ## Load model and processor
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft").cuda()
    
    
    ## Load audio and subtitle data
    audio_input, sample_rate = sf.read(args.audio_dir)
    if (audio_input.shape[-1] <= 2) and (audio_input.shape[1] != 1):
        sound = AudioSegment.from_wav(args.audio_dir)
        sound = sound.set_channels(1)
        sound.export(args.audio_dir, format="wav")
    
        audio_input, sample_rate = sf.read(args.audio_dir)
    
    f = open(args.text_dir, "r")
    sub = csv.reader(f)
    
    lines = []
    for line in sub:
        lines.append(line)
    f.close()
    
    
    ## Split audio and subtitles
    print(f"\n>>> Processing Audio...")
    
    audios = []
    subs = []
    timings = []
    
    for line in lines[1:]:
        
        start_idx = int((int(line[1]) / args.frame_rate) * sample_rate)
        end_idx = int((int(line[2]) / args.frame_rate) * sample_rate)
        
        audios.append(audio_input[start_idx:end_idx])
        subs.append(line[3])
        timings.append([int(line[1]) / args.frame_rate, int(line[2]) / args.frame_rate])
        
    print(f">>> Done...\n")
    
    
    ## Phoneme Optimization
    print(f">>> Start Optimization for:  {len(audios)} chunks...")
    
    preds = []
    for i in range(len(audios)):
        pred = optim_model(args, processor, model, audios[i], sample_rate, subs[i])
        preds.append(pred)
    
    print(f">>> Done...\n")
    
    
    ## Find word boundary
    start_infos = []
    end_infos = []
    duration_infos = []
    interv_infos = []
    
    for i in range(len(preds)):
        
        start_idx = timings[i][0]
        end_idx = timings[i][1]

        start_info, end_info, duration_info, interv_info = find_word(
            args, processor, audios[i], sample_rate,
            preds[i], subs[i], start_idx, end_idx
        )

        start_infos.append(start_info)
        end_infos.append(end_info)
        duration_infos.append(duration_info)
        interv_infos.append(interv_info)
        
    
    np_start_infos = np.array(start_infos)
    np_end_infos = np.array(end_infos)
    np_duration_infos = np.array(duration_infos)
    np_interv_infos = np.array(interv_infos)
    
    np.save(f"./data/start_infos", np_start_infos)
    np.save(f"./data/end_infos", np_end_infos)
    np.save(f"./data/duration_infos", np_duration_infos)
    np.save(f"./data/pause_infos", np_interv_infos)
        
