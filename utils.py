from header import *
import matplotlib.pyplot as plt
import base64

color_table = None
color_index = 0


def get_color_table():
    color_table = []
    for h in range(0, 30):
        hsv = [h*50%180, 255, 255]
        tmp = cv2.cvtColor(np.uint8([[hsv]]), cv2.COLOR_HSV2RGB)[0][0].tolist()
        tmp.append(255)
        color_table.append(tmp)

    return color_table


def color(id=-1):
    global color_table, color_index
    if color_table is None:
        color_table = get_color_table()
    
    if id >= 0:
        color = color_table[id % len(color_table)]
    else:
        color = color_table[color_index]
        color_index = (color_index + 1) % len(color_table)
    return color


def unique_id(input):
    # unique_id = hash(input) & 0xffffffff
    # unique_id = base64.b64encode(input.encode('utf-8'))
    return input.lower()


def filter_input(input):
    # Remove special characters using regex
    cleaned_input = re.sub(r'[^a-zA-Z0-9_ ]', '', input)
    cleaned_input = re.sub(r'_+', '_', cleaned_input)
    cleaned_input = re.sub(r'\s+', ' ', cleaned_input).strip()
    return cleaned_input


def equal_name(a, b):
    return a.lower() == b.lower()


def load_configs():
    with open('config.yaml', 'r') as yfile:
        config = yaml.safe_load(yfile)
    

def parse_annotation_json(file_path):
    required_keys = {
        "info": dict,
        "annotations": dict,
        "categories": dict
    }

    annotations_keys = {'filename': str, 
                'image_width':int,
                'image_height':int,
                'scale':float,
                'bboxes':list,
                'points':list,
                'seg_file':str,
                'instance_ids':list,
                'category_ids':list,
                }
    
    cat_keys = {'name':str,
                'color':list,
                'type':str,
                }
    
    if not os.path.exists(file_path):
        print('Cannot find an annotations.json in {}'.format(file_path))
        return None
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        for key, expected_type in required_keys.items():
            if key not in data:
                print(f"Error: Key '{key}' is missing in the JSON.")
                return None
            elif not isinstance(data[key], expected_type):
                print(f"Error: Key '{key}' has an invalid type. Expected {expected_type}, but got {type(data[key])}.")
                return None
    
        info = data["info"]
        cat = data["categories"]
        annotations = data["annotations"]
        if not isinstance(info['task_type'], str)  or info['tool_created'] != "picA":
            print(f"Error: This is not picA's annotation file.")
            return None

        for id, value in cat.items():
             for key, expected_type in cat_keys.items():
                if key not in value:
                    print(f"Error: Key '{key}' is missing in the JSON.")
                    return None
                elif not isinstance(value[key], expected_type):
                    print(f"Error: Key '{key}' has an invalid type. Expected {expected_type}, but got {type(cat[key])}.")
                    return None
    
        for id, value in annotations.items():
            for key, expected_type in annotations_keys.items():
                if key not in value:
                    print(f"Error: Key '{key}' is missing in the JSON.")
                    return None
                elif not isinstance(value[key], expected_type):
                    print(f"Error: Key '{key}' has an invalid type. Expected {expected_type}, but got {type(annotations[key])}.")
                    return None
        
            data["annotations"][id]['seg'] = None

    except Exception as e:
        print(f"Error: Invalid JSON file.")
        return None
    
    return data


def export_2_coco_format(_metadata):
    pass


def parse_from_coco_format(_metadata):
    pass


def record(history, new):
    if len(history) == 0 or not np.array_equal(history[-1], new):
        history.append(new)
    MAX_HISTORY = 50
    if len(history) > MAX_HISTORY:
        history.pop(0)

def convert_cocomask_to_1chmask(cocomask):
    r = cocomask[:, :, 0]
    g = cocomask[:, :, 1]
    b = (cocomask[:, :, 2]/50.0).astype(int)
    b[b>=5] = 4
    onechmask = r + g*256 + b
    return onechmask.astype(np.uint16)


def convert_1chmask_to_cocomask(onechmask):
    cp_mask = onechmask * 1
    cp_mask[cp_mask<=4] = 0
    onechmask[onechmask>4]=0
    r = cp_mask % 256
    g = cp_mask / 256
    b = onechmask * 50
    cocomask =  np.stack([r, g, b], axis=2)
      
    return cocomask.astype(int)



default_license = [
        {
            "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
            "id": 1,
            "name": "Attribution-NonCommercial-ShareAlike License"
        },
        {
            "url": "http://creativecommons.org/licenses/by-nc/2.0/",
            "id": 2,
            "name": "Attribution-NonCommercial License"
        },
        {
            "url": "http://creativecommons.org/licenses/by-nc-nd/2.0/",
            "id": 3,
            "name": "Attribution-NonCommercial-NoDerivs License"
        },
        {
            "url": "http://creativecommons.org/licenses/by/2.0/",
            "id": 4,
            "name": "Attribution License"
        },
        {
            "url": "http://creativecommons.org/licenses/by-sa/2.0/",
            "id": 5,
            "name": "Attribution-ShareAlike License"
        },
        {
            "url": "http://creativecommons.org/licenses/by-nd/2.0/",
            "id": 6,
            "name": "Attribution-NoDerivs License"
        },
        {
            "url": "http://flickr.com/commons/usage/",
            "id": 7,
            "name": "No known copyright restrictions"
        },
        {
            "url": "http://www.usa.gov/copyright.shtml",
            "id": 8,
            "name": "United States Government Work"
        }
    ]


help_info = 'PicA is an AI-powered image annotation\ntool. If you encounter any issues, please\ndon\'t hesitate to ask on its GitHub page.'
github_url = 'https://github.com/pengyuchu/picA'

version = '0.1.1'