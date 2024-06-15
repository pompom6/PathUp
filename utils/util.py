import os

def walk_dir(data_dir, file_types):
    path_list = []
    for dirpath, dirnames, files in os.walk(data_dir):
        for f in files:
            for this_type in file_types:
                if not f.lower().startswith('._'):
                    if f.lower().endswith(this_type) or f.endswith(this_type):
                        path_list.append(os.path.join(dirpath, f))
                        break
    return path_list

cls_ids = {
    'Normal':0,
    'Pathological Benign':1,
    'Usual Ductal Hyperplasia':2,
    'Flat Epithelial Atypia':3,
    'Atypical Ductal Hyperplasia':4,
    'Ductal Carcinoma in Situ':5,
    'Invasive Carcinoma':6,
}

def get_cls_id(ss):
    ids = ss.split(',')
    class_id = cls_ids[ids[0]]
    if ids[1] == 'overview':
        class_id = class_id + 6
    return class_id







