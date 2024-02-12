# Server & Client

When running a client and a server, make sure to launch the server first, otherwise the client will return an error. 
A server instance can handle several client connections one after the other during its execution, 
but multiple simultaneous clients are not possibles because it would mess with the other workers' conversational contexts.

By default, the communications between client and server are not encrypted, you can add symmetric Fernet encryption 
if you want (recommended if not using ssh tunneling or localhost). See encryption section below for more details.


## Server
### Description

A server worker will create a websocket server, and wait for client connections. When a client is connected, 
all messages received are transmitted to the next local worker, each message received from local workers are sent to 
the client.

### Config example with default parameters
```yaml
socket_server:
  key: "file:.key"    # can be either a file path or string, optional (see encryption section below)
  host: 127.0.0.1
  port: 8080
```

## Client
### Description

Client worker that will connect to a server. Send all input messages to server, and relays server response to the next 
local worker.

### Config example with default parameters
```yaml
socket_client:
  key: "file:.key"    # can be either a file path or string, optional (see encryption section below)
  host: 127.0.0.1
  port: 8080
```

## Encryption

TODO