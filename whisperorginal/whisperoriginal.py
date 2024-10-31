import whisper
import time

model = whisper.load_model("large")
start = time.time()
result = model.transcribe("two.wav")
print(result["text"])
end = time.time()
print("Duration: ", end-start)