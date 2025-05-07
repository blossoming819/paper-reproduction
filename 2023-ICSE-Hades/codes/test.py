# import torch
#
# a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
# print(a.mean(axis=0))
# print(a.mean(axis=1))
#
# tensor([2., 3.])
# tensor([1.5000, 3.5000])

import os
import pickle
data_dir = '../data/chunk_10'
with open(os.path.join(data_dir, "unlabel.pkl"), "rb") as fr:
    chunks = pickle.load(fr)

def extract_schema(data):
    schema ={}
    schema['type'] = str(type(data))
    if hasattr(data,'__dict__'):
        schema['attributes'] = list(data.__dict_.keys())
    elif isinstance(data,dict):
        schema['attributes'] = list(data.keys())
    return schema
#示例用法
schema_info = extract_schema(chunks)
print(schema_info)

with open('../test/unlabel.txt', 'a') as f:
    for item in chunks:
        f.write(str(chunks[item]))