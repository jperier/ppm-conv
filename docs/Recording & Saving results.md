# Recording and saving results

## Recording audio
### Description

The recording worker can be used to record the audio of a session. It will buffer audio chunks for a time before 
saving to a file. By default, it will record to a .wav file every minute (60 chunks of 1sec). 
The recorded files can be found in the `recording/` directory, with timestamps as names.

### Config example with default parameters
```yaml
recording:
  sample_rate: 16000
  buffer_size: 60       # Number of chunks before saving to file
```

## Saving transcripts and commands

### Description

You can save the transcription and the commands produced by the system to files. 
You can save it as a simple text file with timestamps or more detailed json files:

### Config example with default parameters
```yaml
transcript_to_file:
  save_dir: 'transcript/'
  save_details: false       # true will save json format as well as text.
```