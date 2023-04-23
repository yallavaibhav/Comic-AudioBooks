import sys
sys.path.append("Real-Time-Voice-Cloning")
from IPython.display import Audio
from pydub import AudioSegment
from IPython.utils import io
from synthesizer.inference import Synthesizer
from encoder import inference as encoder
from vocoder import inference as vocoder
from pathlib import Path
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import cv2
import moviepy.editor as mymovie
from moviepy.editor import ImageClip, AudioFileClip, CompositeVideoClip, VideoFileClip, concatenate_videoclips

pred0 = ['testers/Venomnibus-v01-0006_jpg.rf.85d0de455b918337d0b360014e39b800 (1).jpg',
         'testers/Venomnibus-v01-0006_jpg.rf.85d0de455b918337d0b360014e39b8002.jpg',
         'testers/Venomnibus-v01-0006_jpg.rf.85d0de455b918337d0b360014e39b8003.jpg',
         'testers/Venomnibus-v01-0006_jpg.rf.85d0de455b918337d0b360014e39b8004.jpg',
         'testers/Venomnibus-v01-0006_jpg.rf.85d0de455b918337d0b360014e39b8005.jpg']
pred1 = pd.read_csv('testers/Venom (2).csv')

def speech(pred):

    encoder_weights = Path("encoder.pt")
    vocoder_weights = Path("vocoder.pt")
    syn_dir = Path("synthesizer.pt")
    encoder.load_model(encoder_weights)
    synthesizer = Synthesizer(syn_dir)
    vocoder.load_model(vocoder_weights)

    charcternames = {47.0: 'audio/1455-134435-0000.flac', 65.0: 'audio/2159-179154-0000.flac', 69.0: 'audio/481-123719-0000.flac', 70.0: 'audio/4088-158077-0000.flac', 200.0: 'audio/26-495-0000.flac'}
    df = pred[0]
    print(df)

    def speech_generate(name, text, panelz, i):
        in_fpath = Path(charcternames[name])
        reprocessed_wav = encoder.preprocess_wav(in_fpath)
        original_wav, sampling_rate = librosa.load(in_fpath)
        preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)
        embed = encoder.embed_utterance(preprocessed_wav)
        with io.capture_output() as captured:
            specs = synthesizer.synthesize_spectrograms([text], [embed])
        generated_wav = vocoder.infer_waveform(specs[0])
        generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")
        audio_path = 'generated_audio/my_audio_file_{0}_{1}.wav'.format(panelz, i)
        sf.write(audio_path, generated_wav, synthesizer.sample_rate)
        return audio_path

    y = []
    x = []
    panelz = 1
    print(len(df))
    for i in range(len(df)):
        print(df.iloc[i][1])
        if panelz == df.iloc[i][0]:
            x.append(speech_generate(df.iloc[i][1], df.iloc[i][7], panelz, i))
        else:
            y.append(x)
            x = []
            panelz += 1
            x.append(speech_generate(df.iloc[i][1], df.iloc[i][7], panelz, i))
    y.append(x)

    print(y)

    # Load the audio file
    panell = 1
    duration_seconds = []
    audioclips = []
    for i in y:
      print("This",i)
      audioo = None
      audioo = AudioSegment.from_file(i[0])
      for j in range(1,len(i)):
        print("hii", i[j])
        audioo += AudioSegment.from_file(i[j])
      tt = "generated_audio/combined_audio{0}.wav".format(panell)
      audioo.export("generated_audio/combined_audio{0}.wav".format(panell), format="wav")
      duration_seconds.append(len(audioo) / 1000.0)
      audioclips.append(tt)
      panell+=1

    img = pred[1]

    imgx=[]
    for i in img:
      img1 = cv2.imread(i)
      resize_image = cv2.resize(img1,(640,640))
      im_rgb = cv2.cvtColor(resize_image, cv2.COLOR_BGR2RGB)
      imgx.append(im_rgb)

    zz = []
    for i in range(len(audioclips)):
        image = ImageClip(imgx[i])
        # Load the audio file
        # audio = AudioFileClip(audioclips[i])
        audio = mymovie.AudioFileClip(audioclips[i])
        print(audio)
        # Create a video clip by combining the image and audio clips
        video = CompositeVideoClip([image.set_duration(duration_seconds[i])])
        videe = video.set_audio(audio)
        # Export the video to a file
        file_video_path = "generated_audio/outputfilee{0}.mp4".format(i)
        videe.write_videofile(file_video_path, fps=24)
        zz.append(file_video_path)

    clips = []
    for i in range(len(zz)):
      print(i)
      clip = VideoFileClip(zz[i])
      clips.append(clip)
    print(clips)
    final_clip = concatenate_videoclips(clips)

    # final_clip.write_videofile("output12.mp4")
    final_clip.write_videofile("static/final/outputpl1.mp4",fps=24, codec='libx264', audio_codec='aac', remove_temp=True)

    return 'Success'

