import json
import pickle

filename = '../visual-genome/objects_2.json'
with open(filename, 'r') as f:
        objects = json.load(f)

objs = [[(o['object_id'], o['names']) for o in objects[i]['objects']] for i in range(len(objects))]

obs = {}
for o in objs:
    for oo in o:
        obs[oo[0]] = oo[1]
        
ids_objects = {}
for o in objects:
    ids_objects[o['image_id']] = []
    for oo in o['objects']:
        ids_objects[o['image_id']].append(oo['names'])
        
        
file = open('../visual-genome/ids_objects.npy', 'wb')
pickle.dump(ids_objects, file)
file.close()        