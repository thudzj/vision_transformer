import os
import xml.etree.ElementTree as ET
import shutil

p = '/home/ubuntu/ILSVRC/Data/CLS-LOC/val/'

for f in os.listdir(p):
    if not f.endswith('JPEG'):
        continue
    path_img = os.path.join(p, f)
    path_anno = path_img.replace('Data', 'Annotations').replace('JPEG', 'xml')

    file = open(path_anno)
    tree = ET.parse(file)
    root = tree.getroot()

    clss = []
    for obj in root.iter('object'):
        cls = obj.find('name').text
        clss.append(cls)
    clss = list(set(clss))

    if len(clss) > 1:
        print(path_anno)
        assert False

    if not os.path.exists(p + cls):
        os.makedirs(p + cls)

    shutil.copy(path_img, p + cls + '/' + f)

