#!/usr/bin/env python3
import base64
import sys
import os
import yaml
import pickle
import numpy

def decode(s: str) -> str:
  s = s.replace("\n", "")
  b = base64.b64decode(s)
  obj = pickle.loads(b)

  return str(obj)



# if __name__ == "__main__":
#   command = sys.argv[1]
#   argument = sys.argv[2]
#   functions = {
#     "decode": decode,
#     "encode": encode,
#   }
#   print(functions[command](argument))

if __name__ == "__main__":
  command = sys.argv[1]
  argument = os.environ["INPUT"]
  functions = {
    "decode": decode,
  }
  output = functions[command](argument)
  print(yaml.dump({"output": output}))


# d = decode("b'\x80\x03G@WA\x99\x99\x99\x99\x9a.'")
# print(d)
# print(type(d))