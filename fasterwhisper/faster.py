import time
from faster_whisper import WhisperModel


model = WhisperModel("large", device="cpu")  
start = time.time()
result = model.transcribe("two.wav")
segments = list(result[0]) 
for segment in segments:
    print(segment.text)
end = time.time()
print("Duration: ", end - start)