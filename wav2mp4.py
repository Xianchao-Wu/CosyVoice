from moviepy.editor import AudioFileClip, ImageClip

def to_mp4(imgfn, wavfn):
    audio = AudioFileClip(wavfn)
    image = ImageClip(imgfn, duration=audio.duration)

    video = image.set_audio(audio)

    outfn = wavfn.replace('.wav', '.mp4')
    video.write_videofile(outfn, fps=24)

for tag in ['fine_grained_control_0', 'instruct_0', 'zero_shot_0']:
    imgfn = tag + '.png'
    wavfn = tag + '.wav'

    to_mp4(imgfn, wavfn)


