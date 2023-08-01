import torch
import argparse
import math
import soundfile as sf

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





if __name__ == "__main__":
    
    
    ## Set arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--steps', type=int, default=50)
    args = parser.parse_args()
    
    
    ## Set parameters
    initial_lr = 8e-6
    threshold = 0.5
    
    
    
    # load model and processor
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft").cuda()
    
    
    
    ## Process input audio
    librispeech_samples_ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")

    # load audio
    audio_input, sample_rate = sf.read(librispeech_samples_ds[0]["file"])

    # pad input values and return pt tensor
    input_values = processor(audio_input, sampling_rate=sample_rate, return_tensors="pt").input_values.cuda()
    
    
    
    print(len(audio_input))
    print(input_values.size())
    
    
    
    
    ## Set target transcription
    target_transcription = "A MAN SAID TO THE UNIVERSE I EXIST"
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
        
    
    ## Get final label
    print("\n")
    print(f"Original phoneme label:    {labels.tolist()[0]}")
    
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    

    pred_labels = []
    for i in range(predicted_ids.size(1)):
        temp = predicted_ids[:,i]
        if int(temp) != 0:
            pred_labels.append(int(temp))
            
            
            
    print(f"Predicted phoneme label:   {pred_labels}")
    
    print(f"\n\n Predicted phoneme sequence:")
    print(predicted_ids)
    
    print(predicted_ids.size())