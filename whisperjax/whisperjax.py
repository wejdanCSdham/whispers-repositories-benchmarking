
import time
import jax
from whisper_jax import FlaxWhisperPipline


pipeline = FlaxWhisperPipline("openai/whisper-large-v2")  

start = time.time()
result = pipeline("two.wav")
print(result["text"])
end = time.time()
print("Duration:", end - start)