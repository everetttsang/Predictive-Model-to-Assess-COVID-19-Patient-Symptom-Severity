#!/usr/bin/env python
# coding: utf-8


import json

def remove_error_info(d):
    if not isinstance(d, (dict, list)):
        return d
    if isinstance(d, list):
        return [remove_error_info(v) for v in d]
    return {k: remove_error_info(v) for k, v in d.items()
            if k not in {'metadata', 'start', 'end', 'ref_end', 'ref_spans', 'cite_spans', 'bib_entries'}}




filenames = open('files.txt', 'r')
Lines = filenames.readlines()
#print(Lines)
worklist=[]
for line in Lines:
    worklist.append(line.strip())

print(worklist)

for item in worklist:
    #print("Opening: " + item)
    with open(item) as f:
        data = json.load(f)

    data = remove_error_info(data)
    json_string = json.dumps(data, indent = 4, sort_keys=True)
    print(json_string)

    with open(item, 'w') as json_file:
      json.dump(data, json_file, indent =4, sort_keys=True)
print("Done")
