#!/usr/bin/env python3
import json
import base64
import sys
import os
import yaml


# json_data = "{\"master\": 1,\"method\":\"master\",\"kwargs\": {\"ids\" : 1 } }"
# json_data = "{'master': 1,'method':'master', 'kwargs': {'ids' : 1 } }"


def input_func(json_data: str) -> str:
    json_data = json_data.replace("\'", "\"")
    json_data=json.loads(json_data)
    json_bytes = json.dumps(json_data).encode()
    data_format_bytes = 'json'.encode()
    serialized_input = data_format_bytes + b'.' + json_bytes
    prepared_input =  base64.b64encode(serialized_input).decode('UTF-8')
    return prepared_input


# if __name__ == "__main__":
#     command = sys.argv[1]
#     argument = sys.argv[2]
#     functions = {
#         "input_func": input_func
#     }
#     print(functions[command](argument))

if __name__ == "__main__":
  command = sys.argv[1]
  argument = os.environ["INPUT"]
  functions = {

    "input_func": input_func

  }
  output = functions[command](argument)
  print(yaml.dump({"output": output}))

# print(input_func(json_data))