import pandas as pd
import csv

# t = pd.read_csv("./local/mnist_train.csv", header=None)
#
#
# #create header
#
# list = [*range(0, 785, 1)]
# # print(list)
# t.to_csv("./local/mnist_train_header.csv", header=list, index=False)

# t = pd.read_csv("./local/mnist_test.csv", header=None)
#
#
# #create header
#
# list = [*range(0, 785, 1)]
# # print(list)
# t.to_csv("./local/mnist_test_header.csv", header=list, index=False)
#



t_1 = pd.read_csv("./local/mnist_train_header.csv")
print(len(t_1))
t_2 = pd.read_csv("./local/mnist_test_header.csv")
print(len(t_2))

#split training file for multiple nodes

f = [t_1,t_2]
r = pd.concat(f)
print(len(r))
# print(r[:5])
df1=r[:35000]
df2=r[35000:]

print(len(df1))
print(len(df2))

df1.to_csv("./local/mnist_train_test_header_1.csv", index=False)
df2.to_csv("./local/mnist_train_test_header_2.csv", index=False)


#converting input
# import json
# import base64

# json_data =  '{"master": 1,"method":"master", "kwargs": { "column_name": "age"}}'
#
# print(type(json_data))
# json_data = json.loads(json_data)
# print(type(json_data))
# json_bytes = json.dumps(json_data).encode()
# data_format_bytes = 'json'.encode()
# print(type(json_bytes))
# print(type(data_format_bytes))
# serialized_input = data_format_bytes + b'.' + json_bytes
# prepared_input =  base64.b64encode(serialized_input).decode('UTF-8')
# print(type(prepared_input))
# print(prepared_input)