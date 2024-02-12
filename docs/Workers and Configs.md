# Workers & Configurations

## Workers

This framework is based on multiprocessing. Each process spawned during execution is called a Worker. 
It has a specific task to accomplish (e.g. Streaming audio from mic, transcribing audio to text, recording audio).
Each worker can receive and send data to other workers, it will do its task in real-time as it generates or 
receives data.

## Configuration

Configuration files are used to define :
 - The workers that you want to spawn
 - The parameters of each worker
 - The communication order between workers

The program uses YAML files, which follow this format:
```yaml
global:
  global_param_1: value     # can be used by all workers if they need this parameter

worker_1:                   # This worker can generate something (e.g. get audio from microphone)
  str_param_1: value        # Worker-specific parameters
  str_param_2: value
  to: worker_2              # 'to' keyword is used to set the worker(s) to which the result will be sent

worker_2:                   # This worker will receive what worker_1 has generated
  param_boolean_1: true      
  to: [worker_3, worker_4]  # This process sends a copy of its result to 2 workers.

worker_3:                   # This worker does not have 'to', so it does not send its data to any other worker
  embed_params:             # in some cases parameters can be a mapping (python dict)
    int_param: 0
    float_param: 0.6
  null_param: ~             # null (None in python) is represented as ~)
    
worker_4: ~                 # This worker will have all default parameters
  
```

## Example

For instance, if you use microphone streaming, which sends audio to a text-to-speech, and then you print 
the transcribe text, you will have 3 workers (Device stream, TTS & print). 
In this case you will want the communication to be (Device stream -> TTS -> print).
This is an example of a configuration file:

```yaml
global:
  sample_rate: 16000        # This parameter is used by device_stream and asr

device_stream:
  segment_length: 16000     # Will generate 1s-long chunks of audio (at 16k sample rate)
  to: asr                   # sends data to asr
  
asr:
  model_size: tiny      
  to: print                 # After it has transcribed audio, sends text to print
  
print: ~                    # Just prints text, does not send any data
```
 
If you also want to record the audio that is streamed, you can do so by adding a worker that does it:

```yaml
global:
  sample_rate: 16000        # This parameter is used by device_stream, asr as well as recording

device_stream:
  segment_length: 16000     
  to: [asr, recording]      # Will send a copy of audio to asr and to recording
  
asr:
  model_size: tiny      
  to: print
  
print: ~

recording:
  buffer_size: 60           # will buffer 60 audio chunks before saving to file
```


## Going a bit further: messages and commands

TODO
