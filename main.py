# 商业转载请联系作者获得授权，非商业转载请注明出处。
# For commercial use, please contact the author for authorization. For non-commercial use, please indicate the source.
# 协议(License)：署名-非商业性使用-相同方式共享 4.0 国际 (CC BY-NC-SA 4.0)
# 作者(Author)：lukeewin
# 链接(URL)：https://blog.lukeewin.top/archives/faster-whisper
# 来源(Source)：lukeewin的博客


# download model
# large-v3模型：https://huggingface.co/Systran/faster-whisper-large-v3/tree/main
# large-v2模型：https://huggingface.co/guillaumekln/faster-whisper-large-v2/tree/main
# large-v2模型：https://huggingface.co/guillaumekln/faster-whisper-large-v1/tree/main
# medium模型：https://huggingface.co/guillaumekln/faster-whisper-medium/tree/main
# small模型：https://huggingface.co/guillaumekln/faster-whisper-small/tree/main
# base模型：https://huggingface.co/guillaumekln/faster-whisper-base/tree/main
# tiny模型：https://huggingface.co/guillaumekln/faster-whisper-tiny/tree/main

from faster_whisper import WhisperModel

import os
model_size = "medium"
#model_size = "large-v3"
#model_size = "tiny"

#path = r"D:\Works\Python\Faster_Whisper\model\small"
#path = r"C:\tony\dev\faster.whisper.01\model\faster-whisper-tiny"
path = r"C:\tony\dev\faster.whisper.01\model\faster-whisper-medium"
transcribe_file=r"C:\tony\dev\faster.whisper.01\transcribe.txt"
audio_file="C:\\tony\\dev\\faster.whisper.01\\short-1.m4a"

# Run on GPU with FP16
#model = WhisperModel(model_size_or_path=path, device="cuda", local_files_only=True)

# or run on GPU with INT8
# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8
model = WhisperModel(model_size, device="cpu", compute_type="int8")

#segments, info = model.transcribe("C:\\tony\\dev\\faster.whisper.01\\short-1.m4a", beam_size=5, language="zh",vad_filter=True, vad_parameters=dict(min_silence_duration_ms=1000))
segments, info = model.transcribe(audio_file, beam_size=5, language="zh",
                                  vad_filter=True, vad_parameters=dict(min_silence_duration_ms=1000))

print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

if os.path.exists(transcribe_file):
  os.remove(transcribe_file)

f = open(transcribe_file, "a",encoding='UTF-8')
for segment in segments:
    print("[%.2fs -> %.2fs] %s\n" % (segment.start, segment.end, segment.text))
    f.write("[%.2fs -> %.2fs] %s\n" % (segment.start, segment.end, segment.text))
f.close()

print("Process Completed")
