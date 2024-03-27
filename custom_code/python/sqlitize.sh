#!/bin/bash

db=/tmp/data.db

rm $db

sqlite-utils insert $db lessons typing-data.json --flatten

sqlite-utils convert $db lessons histogram \
'import json
cols = {}
for i in json.loads(value):
    ltr = chr(i["codePoint"]) if i["codePoint"] != 32 else "spc"
    cols[f"{ltr}_hitCount"] = i["hitCount"]
    cols[f"{ltr}_missCount"] = i["missCount"]
    cols[f"{ltr}_timeToType"] = i["timeToType"]
return cols' --multi

sqlite-utils transform $db lessons --drop histogram

sqlite-utils vacuum $db

# ls -al data.db

python sqlite_parse.py
