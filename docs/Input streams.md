# Input Streams

We currently support 2 different input audio streaming methods: Microphone (device) and File.

## Device Streaming
### Description

This worker will stream audio chunks from an audio device. It uses a `torchaudio.io.StreamReader` to do so.
You can chance the audio device, audio driver, chunk size and sample rate.

### Config example with default parameters
```yaml
device_stream:
  src_device: "hw:0,0"    # This is used to specify the device that you want to use
  audio_format: "alsa"    # This is the default Ubuntu audio driver, you may need to change it
  sample_rate: 16000      # The sample rate to use 16k is better for use with OpenAI Whisper model
  segment_length: 16000   # The size of each audio chunk that will be produce, by default 1sec with 16k sample rate
  command_mode: "conv"    # values: "conv" or "transcribe"
```
**Further notes:**
- `command_mode`: more details to be added # TODO


## File Streaming
### Description

This worker can be used to stream the audio of files. 
You can use it to stream a single file or all files in a directory.

TODO Preciser resample, taille des fichiers en limitation

### Config example with default parameters
```yaml
file_stream:
  path: "path/to/data"    # File or directory to stream from
  pause_time: 0.4         # Time to wait between sending chunks, can be used to not overflow queues
  segment_length: 16000   # Length of each chunk
  use_file_time: true     # Use file time codes (true) or real time (false)
  command_mode: "conv"    # values: "conv" or "transcribe"
```
