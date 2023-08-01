from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from datasets import load_dataset
import torch

import warnings
import soundfile as sf


# load model and processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")


# load dummy dataset and read soundfiles
ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
 

audio_input, sample_rate = sf.read(ds[0]["file"])
input_values = processor(audio_input, sampling_rate=sample_rate, return_tensors="pt").input_values
    
    
# print(len(ds[0]["audio"]["array"]))
    

# tokenize
# input_values = processor(ds[0]["audio"]["array"], return_tensors="pt", sampling_rate=16000).input_values

# print(input_values.size())


# retrieve logits
with torch.no_grad():
    logits = model(input_values).logits
    
    
# print(logits.size())
    
    
# take argmax and decode
predicted_ids = torch.argmax(logits, dim=-1)
# print(predicted_ids)


# print(predicted_ids.size())


transcription = processor.batch_decode(predicted_ids)
print(len(transcription[0].split(" ")))


target_transcription = "A MAN SAID TO THE UNIVERSE I EXIST"

# encode labels
with processor.as_target_processor():
    labels = processor(target_transcription, return_tensors="pt").input_ids
    

print(labels.size(1))


# with torch.no_grad():
#     gt_logits = model(input_values, labels=labels).logits
    
# gt_ids = torch.argmax(gt_logits, dim=-1)

# print(processor.batch_decode(gt_ids))
    
    
# print(labels)
# print(labels.size())
    
    
    
# print(labels)


# # compute loss by passing labels
# loss = model(input_values, labels=labels).loss
# loss.backward()

# print(loss)



# with torch.no_grad():
#     logits = model(input_values).logits
    
# predicted_ids = torch.argmax(logits, dim=-1)

