import json

json_file = open("parameters.json","r")
y=json.load(json_file)
json_file.close()
num_layer=int(y['layer'])
num_units=int(y['units'])
num_epoch=int(y['epoch'])
y['layer']+=1
y['epoch']+=1
print(y)
y=json.dumps(y)
json_file = open("parameters.json","w")
json_file.write(y)
json_file.close()
