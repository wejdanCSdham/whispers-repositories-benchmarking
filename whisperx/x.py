import time
import whisperx

device = "cpu" 
compute_type = "int8" if device == "cpu" else "float16"  
model = whisperx.load_model("large", device, compute_type=compute_type)  

audio = "two.wav"

start = time.time()
result = model.transcribe(audio)
print(result["segments"]) 
end = time.time()
print("Duration: ", end - start)
