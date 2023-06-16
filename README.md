# UPDATE 2023: This project is completely outdated. Thanks to the progress in Deep Learning, we now have new powerful tools!

My new unpublished project works exclusively with YouTube URLs and produces a final video with a new English audio track. The process involves the following steps:

1. Downloading the video using `yt_dlp`.
2. Extracting the audio to WAV format.
3. Utilizing WhisperX for text recognition.
4. Saving the recognized text as `.STR` and a special `.TXT` format for further processing.
5. Correcting errors in the text using `chatGPT` with a specific prompt.
6. Translating the text to English using `chatGPT` with a dedicated prompt based on the video's topic context.
7. Generating WAV files for each subtitle block using TTS (`TorToiSe`).
8. Processing all these WAV files with WhisperX to compare the recognized text with the subtitle text.
9. Concatenating all WAV files into a `FINAL.WAV` file, arranging them based on target time labels on the timeline.
10. Rendering the `FINAL.MP4` file using `moviepy`, combining the original video with the new audio track and background music.
11. Generating XX thumbnails to choose from, using random video frames and adding text using Pillow.

Therefore, the input for this project is a URL, and the output is the `FINAL.MP4` file.

## Therefore, the text below is merely historical and can be disregarded

---
---
---

## Audio track generator from subtitles on YouTube

## The main use case

## You have video with non-English audio, but you have English subtitles (or going to prepare). Now you are ready to generate new English audio-track for this video

![use-case](/img/use-case.jpg)

## Set your parameters in 'Setting' (like video ID from URL) and run this jupyter notebook step-by-step skipping optional cells (like DeepSpeech testing for generated audio)

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
