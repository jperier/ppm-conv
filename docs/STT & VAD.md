# Speech to text & Voice activity detection

## Speech to Text
### Description

We use OpenAI Whisper as a Speech To Text model, our code uses the [official repo](https://github.com/openai/whisper).

By default, our system uses streaming with pretty large chunks (~1 second), Whisper can work with that, 
but we also add 2 types of techniques to add context to each audio chunk :

- Prefixing some part, or the totality, of the last chunk of audio (with or without the last transcribed text) at each
time step. This will add some context to the beginning of the chunk, enabling better transcription if a sentence is 
seperated in two during streaming. Whisper has a special parameter for prefixing previously transcribed text,
you can choose whether you enable text prefixing on top of audio prefixing or not.

- Buffering audio chunks until a conversational turn end or a silence is detected (so that whisper can transcribe 
the entire turn as one audio chunk, which can lead to better performance on long sentences). This can make processing 
a bit slower depending on your setup.


### Config example with default parameters
```yaml
asr:
  model_size: large-v2          # the model to use, recommended 'large-v2' for performance and 'tiny' for testing
  no_speech_max_prob: 0.7       # threshold of whisper no-speech detection above which chunk is consider not to contain speech
  get_speech_segments: false    # whether you want whisper to guess timestamp for speech segments
  decoding_options: ~           # whisper decoding parameters, more details below
  context_type: ~               # 'prefix', 'buffer' or '~' (null)
  context_options: ~            # context parameters, dependents on the context type used
```

**Further notes:**
- `decoding options` are whisper's decoding parameters 
([see Whisper code for more info](https://github.com/openai/whisper/blob/ba3f3cd54b0e5b8ce1ab3de13e32122d0d5f98ab/whisper/decoding.py#L81)). 
You should be able to use all of the original library's parameters. Here are some options that we find useful in our 
use case :

```yaml
decoding_options:
  language: fr      # If you know the language of your users, forcing ill lead to better results
  fp16: true        # Better computing performance if true, but some CPUs might not be compatible (true by default)
  sample_len: 20    # Maximum number of token decoded, can reduce hallucinations, especially for short audio chunks
  temperature: 0.0  # Temperature at 0 will reduce de random nature of the model
```

- `context_options` are context-type dependent, here are the defaults :
```yaml
context_type: prefix
context_options:
  prefix_size_ratio: 0.1    # Here we only prefix the last 10% of the last audio chunk, 1.0 would be the entire chunk
  prefix_text: false        # Whether to give whisper the previous text generated (true recommended for high ratio sizes)
```
```yaml
context_type: buffer
context_options:
  buffer_size: 10           # Maximum size of the buffer (default 10 audio chunks)
```


## Voice Activity Detection (VAD)

### Description

Voice Activity Detection can detect if a given audio sample contains speech.
We use the [Silero VAD model](https://github.com/snakers4/silero-vad) for this. 

For each audio chunk generted by our system, a VAD worker can scan it by dividing it 
in even smaller chunks and computing an average score. If this score is above a given threshold
the chunk is considered to contain speech. If no speech is detected, the VAD worker will not 
let the given chunk pass to the next worker, thus acting like a filter.

The VAD is also used to detect silences. After `n_silence` chunks without speech detected,
a VAD worker will send a silence signal to subsequent workers.

### Config example with default parameters
```yaml
vad:
  repo_str: 'snakers4/silero-vad'
  model_str: 'silero_vad'
  threshold: 0.3          # average VAD score over the chunk to be considered as speech
  n_silence: 3            # number of chunks without voice to send silence signal
  window_size: 512        # 512 is recommanded for 16k sample rate with default model
  sample_rate: 16000
```