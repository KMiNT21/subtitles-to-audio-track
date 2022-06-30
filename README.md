# Audio track generator.

## The main use case:

## You have video with non-English audio, but you have English subtitles (or going to prepare). Now you are ready to generate new English audio-track for this video.

![use-case](/img/use-case.jpg)

## Set your parameters in 'Setting' and run this jupyter notebook step-by-step skipping optional cells (like DeepSpeech testing for generated audio)

1) Download captions for video id, save forever to local file. (Delete  pickle-file and repeat this step if you need to re-download).

2) Contatenate all texts in captions and then clean it before using sentences tokenizer.

3) Split text to sentences using NLTK (Natural Language Toolkit).

4) Synthesize WAV files for each sentence by  **Mozilla TTS**.

5) Compose new captions from these sentences. Arrange start point of each audio segment by matching  with text in original subtitles. Visualize subtitles before and after arrangement by heat-map image. Rows = minutes. Cols = seconds. Numbers in cells = numbers of audio segment.

   ![subtitles](/img/subtitles.png)

6) Concatenate all audio segments in **final.wav**

Example of subtitles arrangement (another video, where final text was edited and simplified).

Before:
![before](/img/before.png)

After:
![after](/img/after.png)

If correspondend text in original subtitles not found then segment will be moved right while free space found. So you should consider if you need to re-calibrate in video-editor (for cases when too many manual changes in text was done).

It is possible that English audio file will be longer then the original sound-track (you can notice this if there are no changes after arrangement). You can ignore it and fix in editor or try faster TTS model.
