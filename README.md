## Install

Create and activate virtual env if necessary:
```bash
python3 -m venv .venv
source .venv/bin/activate
```
Install requirements:
```bash
pip install -r requirements.txt 
```
You will also need to install **ffmpeg** on your device.


## Use
Activate your venv if you are using one and run `main.py` with the path to the config file you want to use.
```
python main.py configs/<your-config-file> -t <intial-timeout>
```
The -t parameter is an initial timeout (in seconds) for all the workers to be ready (if unspecified it will be 120).
If this is your first time running the framework you might want to increase this, as your device may need to download
the models' parameters.

The software was tested on Ubuntu 22.04 and Python 3.10.


## Docs

See the [`docs/`](docs) folder for the documentation.

## More to come

We will release more functionalities in the near future, to encompass all of the conversational functions of an
embodied conversational agent. 

More documentation on extending this framework will also be posted.
