# sample code of Nemo ASR Serving
import nemo
import nemo.collections.asr as nemo_asr

quartznet = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name="QuartzNet15x5Base-En")

file = ['./an4/wav/my_sample.wav']
for fname, transcription in zip(files,quartnet.transcribe(paths2audio_files=files));
    print(f"Audio in (fname) was recongnized as : (transcription)")
