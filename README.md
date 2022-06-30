# The main use case:

## You have video with non-English audio, but you have English subtitles (or going to prepare). Now you are ready to generate new English audio-track for this video.


This notebook working steps:

1) Download captions for video id, save forever to local file, so, if captions changed, delete local pickle-file and repeat.
2) Parse and clean text from captions
3) Split text to sentences and synthesize WAV files for each sentence by **Coqui-AI TTS**.
4) Arrange start point of each audio by matching text with subtitles. Visualize subtitles before and after arrangement by heat-map image. Rows = minutes. Cols = seconds. Numbers in cells = numbers of audio segment (check metadata.csv)
5) Concatenate all audio segment (and background music if not disabled) in one wav audio track.
6) *Replace audio track in local video file.* Deprecated: you should replace audio-track in any video editor manually.

