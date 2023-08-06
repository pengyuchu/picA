'''
This is the picA's kernel script.

Author: Lucas Chu
Date: July 24, 2023
'''
from header import *
import utils
from utils import color, unique_id
from helper import *
import models
from PIL import Image
from skimage.io import imread
from skimage.draw import polygon
from skimage.segmentation import quickshift, mark_boundaries


############################# Global Variables #############################
g_metadata = None
cur_file_id = None
cur_class = None
cur_instance = 0
cur_ratio = 1.0
last_ratio = cur_ratio
mode = None
task = None
tmp_polygon = []
tmp_bbox = []
pre_mask = None
annotation_start = False
use_superpixel = False
use_ai_model = False
select = None


############################# API #############################
def load_models():
    global task
    print('load_models: ', task)
    if task == 'Object Detection':
        type = 'det'
    elif task == 'Object Counting':
        type = 'count'
    elif task == 'Instance Segmentation':
        type = 'ins'
    elif task == 'Semantic Segmentation':
        type = 'sem'
    elif task == 'Panoptic Segmentation':
        type = 'pan'
    else:
        return

    models.load_models([], type=type)


def predict(image):
    global g_metadata, cur_file_id
    pre = models.predict([image])
    if pre is None: return False

    cls = pre['cls']
    for c in cls:
        if c['name'].lower() not in [i['name'].lower() for i in g_metadata['categories'].values()]:
            new_class(c['name'])
        elif c['type'] != g_metadata['categories'][get_class_id(c['name'])]['type']:
            new_class(c['name']+'_'+c['type'], c['type'])
        
    result = pre['results'][0]

    g_metadata['annotations'][cur_file_id]['bboxes'] = result['bboxes']
    g_metadata['annotations'][cur_file_id]['points'] = result['points']
    if task in ['Instance Segmentation', 'Semantic Segmentation', 'Panoptic Segmentation']:
        seg = result['seg']
        for i, cat_id in enumerate(result['category_ids']):
            cat_id = get_class_id(cat_id)
            if g_metadata['categories'][cat_id]['type'] == 'sem':
                index = g_metadata['annotations'][cur_file_id]['category_ids'].index(cat_id)
                ins_id = g_metadata['annotations'][cur_file_id]['instance_ids'][index]
            elif g_metadata['categories'][cat_id]['type'] == 'ins':
                ins_id = new_id()
                g_metadata['annotations'][cur_file_id]['instance_ids'].append(ins_id)
                g_metadata['annotations'][cur_file_id]['category_ids'].append(cat_id)

            seg[result['instance_ids'][i]==seg] = ins_id

        g_metadata['annotations'][cur_file_id]['seg'] = seg
        g_metadata['annotations'][cur_file_id]['color_seg'] = cvt_seg_2_color(seg)
    else:
        g_metadata['annotations'][cur_file_id]['category_ids'] = [get_class_id(i) for i in result['category_ids']]
        g_metadata['annotations'][cur_file_id]['instance_ids'] = result['instance_ids']

    return True


def new_instance(_class_id):
    global g_metadata, cur_file_id, cur_class, cur_instance, task

    if task in ['Instance Segmentation', 'Semantic Segmentation', 'Panoptic Segmentation']:
        anno = g_metadata['annotations'][cur_file_id]
        indices = [i for i, v in enumerate(anno['category_ids']) if v == _class_id]
        if len(indices) > 0:
            ins_id = anno['instance_ids'][indices[-1]]
        else:
            ins_id = -1

        if g_metadata['categories'][_class_id]['type'] == 'sem':
            cur_instance = ins_id
            return False
        else:
            mask = anno['seg']
            if mask is not None and np.sum(mask[mask==ins_id]) > 0:
                cur_instance = new_id()
                g_metadata['annotations'][cur_file_id]['instance_ids'].append(cur_instance)
                g_metadata['annotations'][cur_file_id]['category_ids'].append(cur_class)
                return True
            else:
                cur_instance = ins_id
                return False
    else:
        return False
        

def is_instance(_class_id):
    global g_metadata, cur_file_id
    try:
        return g_metadata['categories'][_class_id]['type'] == 'ins'
    except ValueError:
        print('error in is_instance: ', _class_id)
        return False


def semantic2instance(_class_id):
    global g_metadata, cur_file_id
    if _class_id in g_metadata['categories']:
        g_metadata['categories'][_class_id]['type'] = 'ins'


def get_class_id(_class_name):
    global g_metadata
    _class_ids = [key for key, value in g_metadata['categories'].items() if value['name'] == _class_name]
    if len(_class_ids) > 0:
        return _class_ids[0]
    else:
        return None
    

def new_class(_name, _type=None):
    global g_metadata, cur_class, task

    _id = unique_id(_name)

    if task == 'Object Detection':
        type = 'rec'
    elif task == 'Object Counting':
        type = 'pts'
    elif task == 'Instance Segmentation':
        type = 'ins'
    elif task == 'Semantic Segmentation':
        type = 'sem'
    elif task == 'Panoptic Segmentation':
        type = 'pan'
    else:
        type = ''

    type = type if _type is None else _type
    g_metadata['categories'][_id] = {
        'name': _name,
        'color': color(),
        'type': type if type not in ['sem', 'pan'] else 'sem',
    }
    cur_class = _id

    if task in ['Instance Segmentation', 'Semantic Segmentation', 'Panoptic Segmentation']:
        g_metadata['annotations'][cur_file_id]['instance_ids'].append(new_id())
        g_metadata['annotations'][cur_file_id]['category_ids'].append(cur_class)

def delete_class_global(class_id):
    global g_metadata
    removed_class = g_metadata['categories'].pop(class_id, None)
    for key in g_metadata['annotations'].keys():
        delete_class_local(class_id, key)


def delete_class_local(class_id, file_id):
    global g_metadata
    cur = g_metadata['annotations'][file_id]
    indices = [i for i, v in enumerate(cur['category_ids']) if v == class_id]
    mask = cur['seg']
    if mask is not None:
        for i in indices:
            print(cur['instance_ids'][i], i)
            rr, cc = np.where(cur['instance_ids'][i]==mask)
            g_metadata['annotations'][file_id]['seg'][rr, cc] = 0
            g_metadata['annotations'][file_id]['color_seg'][rr, cc, :] = [0, 0, 0, 0]


    g_metadata['annotations'][file_id]['instance_ids'] = [cur['instance_ids'][i] for i in range(len(cur['instance_ids'])) if i not in indices]
    g_metadata['annotations'][file_id]['category_ids'] = [cur['category_ids'][i] for i in range(len(cur['category_ids'])) if i not in indices]
    if task == 'Object Detection':
        g_metadata['annotations'][file_id]['bboxes'] = [cur['bboxes'][i] for i in range(len(cur['bboxes'])) if i//4 not in indices]
    elif task == 'Object Counting':
        g_metadata['annotations'][file_id]['points'] = [cur['points'][i] for i in range(len(cur['points'])) if i not in indices]


def on_assign_class(_class_name):
    global g_metadata, cur_class, cur_file_id, select
    if select is not None:
        new_cat_id = get_class_id(_class_name)
        cur_type = g_metadata['categories'][cat_ids[select]]['type']
        new_type = g_metadata['categories'][new_cat_id]['type']
        cat_ids = g_metadata['annotations'][cur_file_id]['category_ids']
        if new_type == 'sem':
            index = cat_ids.index(new_cat_id)
            cur_ins_id = g_metadata['annotations'][cur_file_id]['instance_ids'][select]
            new_ins_id = g_metadata['annotations'][cur_file_id]['instance_ids'][index]
            g_metadata['annotations'][cur_file_id]['instance_ids'][select] = new_ins_id
            seg = g_metadata['annotations'][cur_file_id]['seg']
            rr, cc = np.where(cur_ins_id==seg)
            g_metadata['annotations'][cur_file_id]['seg'][rr, cc] = new_ins_id
            g_metadata['annotations'][cur_file_id]['color_seg'][rr, cc, :] = color(new_ins_id)
        elif cur_type == 'sem' and new_type == 'ins':
            cur_ins_id = g_metadata['annotations'][cur_file_id]['instance_ids'][select]
            new_ins_id = new_id()
            g_metadata['annotations'][cur_file_id]['instance_ids'][select] = new_ins_id
            seg = g_metadata['annotations'][cur_file_id]['seg']
            rr, cc = np.where(cur_ins_id==seg)
            g_metadata['annotations'][cur_file_id]['seg'][rr, cc] = new_ins_id
            g_metadata['annotations'][cur_file_id]['color_seg'][rr, cc, :] = color(new_ins_id)

        g_metadata['annotations'][cur_file_id]['category_ids'][select] = new_cat_id
        return True
    else:
        return False


def on_delete_selected():   
    global select, g_metadata, cur_file_id
    if select is None: return

    cur = g_metadata['annotations'][cur_file_id]
    seg = cur['seg']
    if seg is not None:
        ins_id = cur['instance_ids'][select]
        rr, cc = np.where(ins_id==seg)
        g_metadata['annotations'][cur_file_id]['seg'][rr, cc] = 0
        g_metadata['annotations'][cur_file_id]['color_seg'][rr, cc, :] = [0, 0, 0, 0]

    g_metadata['annotations'][cur_file_id]['instance_ids'] = [cur['instance_ids'][i] for i in range(len(cur['instance_ids'])) if i != select]
    g_metadata['annotations'][cur_file_id]['category_ids'] = [cur['category_ids'][i] for i in range(len(cur['category_ids'])) if i != select]
    
    if task == 'Object Detection':
        g_metadata['annotations'][cur_file_id]['bboxes'] = [cur['bboxes'][i] for i in range(len(cur['bboxes'])) if i//4 != select]
    elif task == 'Object Counting':
        g_metadata['annotations'][cur_file_id]['points'] = [cur['points'][i] for i in range(len(cur['points'])) if i != select]
    
    select = None


def on_file_start(id, _image, args=[]):
    global cur_file_id, cur_instance, g_metadata, cur_ratio, task
    cur_file_id = id
    default_ratio = args[0]
    original_image_size = args[1]
    last_ratio = cur_ratio = 1.0
    cur_instance = 0
    if _image is not None:
        if cur_file_id not in g_metadata['annotations'].keys():
            new_metadata_annotation(cur_file_id)
            
        g_metadata['annotations'][cur_file_id]['scale'] = default_ratio
        g_metadata['annotations'][cur_file_id]['image_width'] = original_image_size[1]
        g_metadata['annotations'][cur_file_id]['image_height'] = original_image_size[0]
        if task in ['Instance Segmentation', 'Semantic Segmentation', 'Panoptic Segmentation']:
            if g_metadata['annotations'][cur_file_id]['seg'] is None:
                g_metadata['annotations'][cur_file_id]['seg'] = np.zeros((_image.shape[0], _image.shape[1]), dtype=np.uint16)
                g_metadata['annotations'][cur_file_id]['color_seg'] = cvt_seg_2_color(g_metadata['annotations'][cur_file_id]['seg'])

            for key, value in g_metadata['categories'].items():
                if key not in g_metadata['annotations'][cur_file_id]['category_ids']:
                        g_metadata['annotations'][cur_file_id]['instance_ids'].append(new_id())
                        g_metadata['annotations'][cur_file_id]['category_ids'].append(key)

                if value['type'] == 'ins':
                    new_instance(key)
 
            if cur_class is not None:
                ins_ids = g_metadata['annotations'][cur_file_id]['instance_ids']
                indices = [i for i, v in enumerate(g_metadata['annotations'][cur_file_id]['category_ids']) if v == cur_class]
                cur_instance = max([ins_ids[i] for i in indices]) if len(indices) > 0 else 0
        else:
            ins_ids = g_metadata['annotations'][cur_file_id]['instance_ids']
            cur_instance = max(ins_ids) if len(ins_ids) > 0 else 0

        preprocess(_image[:, :, :3])


def on_file_end():
    pass
    

def cvt_seg_2_color(_seg):
    color_seg = np.tile(_seg[:, :, np.newaxis], (1, 1, 4))
    id_list = np.unique(_seg).tolist()
    for id in id_list:
        if id == 0: continue
        color_seg = np.where(color_seg==[id, id, id, id], color(id), color_seg)

    return color_seg.astype(np.uint8)


def load_annotations(_id, _annotation, root_path):
    global g_metadata, cur_file_id

    if task in ['Object Detection', 'Object Counting']:
        if len(g_metadata['annotations'][_id]['category_ids']) <= 0 or \
            _id not in g_metadata['annotations']:
            g_metadata['annotations'][_id] = _annotation
    elif task in ['Instance Segmentation', 'Semantic Segmentation', 'Panoptic Segmentation']:
        try:
            if _id not in g_metadata['annotations'] or \
                g_metadata['annotations'][_id]['seg'] is None or \
                np.sum(g_metadata['annotations'][_id]['seg']) == 0:
                g_metadata['annotations'][_id] = _annotation
                g_metadata['annotations'][_id]['seg'] = imread(
                    os.path.join(root_path, 'annotations', 'masks', 
                                _annotation['seg_file']), 
                                cv2.IMREAD_UNCHANGED).astype(np.uint16)
                
                g_metadata['annotations'][_id]['color_seg'] = cvt_seg_2_color(g_metadata['annotations'][_id]['seg'])
        except:
            print('Error: Cannot find the {}\'s seg file. Please check files /path/to/annotations/masks/*.png.'.format(_id))


def init_workspace(root_path, filenames, _metadata):
    global g_metadata, cur_file_id, cur_class, cur_instance, task
    global annotation_start, last_ratio
    global mode, tmp_polygon, tmp_bbox, pre_mask, select

    print('Init workspace: ', root_path)
    g_metadata = None
    cur_file_id = None
    cur_class = None
    cur_instance = 0
    cur_ratio = 1.0
    last_ratio = cur_ratio
    tmp_polygon = []
    tmp_bbox = []
    pre_mask = None
    select = None
    # annotation_start = False

    if _metadata is None:
        create_annotation_metadata(root_path, filenames)
    else:
        import_annotation_metadata(root_path, filenames, _metadata)


def save_workspace():
    global g_metadata, cur_file_id, cur_class, cur_instance, task
    root_path = g_metadata['info']['root_path']
    anno_path = os.path.join(root_path, 'annotations')
    masks_path = os.path.join(anno_path, 'masks')
    color_masks_path = os.path.join(anno_path, 'color_masks')
    if not os.path.exists(anno_path):
        os.makedirs(anno_path)

    if not os.path.exists(masks_path) and task in ['Instance Segmentation', 'Semantic Segmentation', 'Panoptic Segmentation']:
        os.makedirs(masks_path)

    if not os.path.exists(color_masks_path) and task in ['Instance Segmentation', 'Semantic Segmentation', 'Panoptic Segmentation']:
        os.makedirs(color_masks_path)

    with open(os.path.join(anno_path, 'annotations.json'), 'w') as f:
        saved_json = {}
        for key in g_metadata.keys():
            saved_json[key] = copy.deepcopy(g_metadata[key])
            
        for id, item in saved_json['annotations'].items():
            if item['seg'] is not None:
                Image.fromarray(saved_json['annotations'][id]['seg']).save(os.path.join(masks_path, saved_json['annotations'][id]['seg_file']))
                cv2.imwrite(os.path.join(color_masks_path, saved_json['annotations'][id]['seg_file'].split('.')[0] + '.png'), saved_json['annotations'][id]['color_seg'])
                saved_json['annotations'][id]['seg'] = None
                saved_json['annotations'][id]['color_seg'] = None
                saved_json['annotations'][id]['pre_mask'] = None

        json.dump(saved_json, f)
    


def check_multiple_annotation_files():
    pass


def load_annotation_metadata(filename):
    global g_metadata, task

    # load annotation file
    set_task_type()
    if g_metadata is not None:
        check_multiple_annotation_files()
        

def set_task_type():
    global task, annotation_start
    if not annotation_start:
        g_metadata['info']['task_type'] = task
        annotation_start = True


def new_metadata_annotation(filename):
    global g_metadata
    id = filename # + '@' + str(uuid.uuid4())
    g_metadata['annotations'][id] = {'filename': filename, 
                                            'image_width':0,
                                            'image_height':0,
                                            'scale':1.0,
                                            'bboxes':[],
                                            'points':[],
                                            'seg':None,
                                            'color_seg':None,
                                            'seg_file':filename.split('.')[0] + '.png',
                                            'pre_mask':None,
                                            'instance_ids':[],
                                            'category_ids':[],
                                            }
    

def import_annotation_metadata(root_path, filenames, _metadata):
    global g_metadata, task, cur_class
    g_metadata = _metadata
    g_metadata['info']['root_path'] = root_path
    task = g_metadata['info']['task_type']

    try:
        cur_class = list(g_metadata['categories'].keys())[0]
    except:
        print('Empty categories')

    for i, v in g_metadata['annotations'].items():
        load_annotations(i, v, root_path)

    set_task_type()


def create_annotation_metadata(root_path, filenames):
    global g_metadata, task
    g_metadata = {}
    g_metadata['info'] = {}
    g_metadata['info']['description'] = ''
    g_metadata['info']['root_path'] = root_path
    g_metadata['info']['date_created'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    g_metadata['info']['tool_created'] = 'picA'
    g_metadata['info']['task_type'] = ''
    g_metadata['info']['task_status'] = 'in progress'
    g_metadata['categories'] = {}
    g_metadata['annotations'] = {}
    g_metadata['licenses']  = utils.default_license
    set_task_type()
    for filename in filenames:
        new_metadata_annotation(filename)


def new_id(base=0):
    global cur_instance, g_metadata, cur_file_id
    ins_ids = g_metadata['annotations'][cur_file_id]['instance_ids']
    cur_instance = max(ins_ids) if len(ins_ids) > 0 else 0
    cur_instance += base + 1
    return cur_instance


def preprocess(_image):
    global mode, task, g_metadata, cur_file_id, pre_mask
    global use_ai_model, use_superpixel
    if task == 'Object Detection':
        pass
    elif task == 'Object Counting':
        pass
    elif task in ['Instance Segmentation', 'Semantic Segmentation', 'Panoptic Segmentation']:
        if use_superpixel:
            if g_metadata['annotations'][cur_file_id]['pre_mask'] is None:
                pre_mask = quickshift(_image, kernel_size=3, max_dist=6, ratio=0.5)
                g_metadata['annotations'][cur_file_id]['pre_mask'] = pre_mask
            else:
                pre_mask = g_metadata['annotations'][cur_file_id]['pre_mask']
        if use_ai_model:
            pass


def on_select(x, y):
    global mode, select, task
    x, y = [round(x/cur_ratio), round(y/cur_ratio)]
    select_set_flag = False
    if task == 'Object Detection':
        bbox = g_metadata['annotations'][cur_file_id]['bboxes']
        if len(bbox) >= 4:
            bbox = np.reshape(bbox, (-1, 4))
            for i, b in enumerate(bbox):
                if x >= b[0] and x <= b[2] and y >= b[1] and y <= b[3]:
                    select = i
                    select_set_flag = True
                    break

        select = select if select_set_flag else None
        
    elif task == 'Object Counting':
        pts = g_metadata['annotations'][cur_file_id]['points']
        if len(pts) >= 1:
            pts = np.reshape(pts, (-1, 2))
            for i, p in enumerate(pts):
                if abs(p[0] - x) <= 5 and abs(p[1] - y) <= 5:
                    select = i
                    select_set_flag = True
                    break
        
        select = select if select_set_flag else None
        
    elif task in ['Instance Segmentation', 'Semantic Segmentation', 'Panoptic Segmentation']:
        seg = g_metadata['annotations'][cur_file_id]['seg']
        value = seg[y, x]
        if value > 0:  
            select = g_metadata['annotations'][cur_file_id]['instance_ids'].index(value)
        else:
            select = None

    else:
        select = None

    return select


def on_rectangle_end(start_xy, end_xy):
    global g_metadata, cur_file_id, cur_class
    global tmp_bbox
    global cur_ratio
    # print('Rectangle: ', start_xy, end_xy)
    if len(tmp_bbox) == 4:
        bbox = [start_xy[0], start_xy[1], end_xy[0], end_xy[1]] #xyxy
        bbox = [i/cur_ratio for i in bbox]
        g_metadata['annotations'][cur_file_id]['bboxes'].extend(bbox)
        g_metadata['annotations'][cur_file_id]['instance_ids'].append(new_id())
        g_metadata['annotations'][cur_file_id]['category_ids'].append(cur_class)
        tmp_bbox = []   



def on_rectangle(start_xy, end_xy):
    global tmp_bbox
    tmp_bbox = [start_xy[0], start_xy[1], end_xy[0], end_xy[1]] #xyxy


def on_counting_click(x, y):
    global g_metadata, cur_file_id, cur_class
    x/=cur_ratio
    y/=cur_ratio
    pts = g_metadata['annotations'][cur_file_id]['points']
    tmp_result = [i for i in pts if abs(i[0] - x) <= 4 and abs(i[1] - y) <= 4]
    if len(tmp_result) == 0:
        g_metadata['annotations'][cur_file_id]['points'].append([x, y])
        g_metadata['annotations'][cur_file_id]['instance_ids'].append(new_id())
        g_metadata['annotations'][cur_file_id]['category_ids'].append(cur_class)


def on_segmentation_click(x, y):
    global g_metadata, cur_file_id, cur_class, pre_mask
    global cur_instance, cur_ratio
    if pre_mask is None:
        preprocess()
    mask_id = pre_mask[round(y/cur_ratio), round(x/cur_ratio)]
    # record(history, copy.deepcopy(g_metadata['annotations'][cur_file_id]['seg']))
    rr, cc = np.where(pre_mask == mask_id)
    g_metadata['annotations'][cur_file_id]['seg'][rr, cc] = cur_instance  
    g_metadata['annotations'][cur_file_id]['color_seg'][rr, cc, :] = color(cur_instance)


def on_draw(x, y):
    global g_metadata, cur_file_id, cur_class, pre_mask
    global cur_instance, cur_ratio
    if pre_mask is None:
        preprocess()
    mask_id = pre_mask[round(y/cur_ratio), round(x/cur_ratio)]
    rr, cc = np.where(pre_mask == mask_id)
    g_metadata['annotations'][cur_file_id]['seg'][rr, cc] = cur_instance 
    g_metadata['annotations'][cur_file_id]['color_seg'][rr, cc, :] = color(cur_instance)
    


def on_polygon_end():
    global g_metadata, cur_file_id, cur_class, cur_instance
    global task, mode
    global tmp_polygon, cur_ratio
    if len(tmp_polygon) < 3: return
    
    mask = g_metadata['annotations'][cur_file_id]['seg']
    tmp_polygon = (np.asarray(tmp_polygon)/cur_ratio).astype(np.uint16)
    rr, cc = polygon(tmp_polygon[:, 1], tmp_polygon[:, 0], mask.shape)
    g_metadata['annotations'][cur_file_id]['seg'][rr, cc] = cur_instance
    g_metadata['annotations'][cur_file_id]['color_seg'][rr, cc, :] = color(cur_instance)
    tmp_polygon = []


def on_polygon_delete():
    global g_metadata, cur_file_id, cur_class, cur_instance
    global task, mode
    global tmp_polygon, cur_ratio
    if len(tmp_polygon) < 3: return

    mask = g_metadata['annotations'][cur_file_id]['seg']
    tmp_polygon = (np.asarray(tmp_polygon)/cur_ratio).astype(np.uint16)
    rr, cc = polygon(tmp_polygon[:, 1], tmp_polygon[:, 0], mask.shape)
    g_metadata['annotations'][cur_file_id]['seg'][rr, cc] = 0
    g_metadata['annotations'][cur_file_id]['color_seg'][rr, cc, :] = [0, 0, 0, 0]
    tmp_polygon = []


def on_polygon(x, y):
    global tmp_polygon, cur_ratio
    tmp_polygon.append((round(x), round(y)))


# Main render function
def render_annotation(original_image, image, show_all=True, show_bbox=False, stroke=1):
    global g_metadata, cur_file_id, cur_class, task, pre_mask
    global tmp_polygon, tmp_bbox
    global last_ratio, cur_ratio

    resized_image = None
    if last_ratio != cur_ratio:
        last_ratio = cur_ratio
        image = cv2.resize(original_image, (int(original_image.shape[1]*cur_ratio), int(original_image.shape[0]*cur_ratio)))        
        resized_image = image
        
    if not annotation_start: return image, resized_image

    _image =image.copy()
    cats = g_metadata['categories']
    cat_ids = g_metadata['annotations'][cur_file_id]['category_ids']
    ins_ids = g_metadata['annotations'][cur_file_id]['instance_ids']
    if task == 'Object Detection':
        bbox = g_metadata['annotations'][cur_file_id]['bboxes']
        if len(bbox) >= 4:
            bbox = (np.reshape(bbox, (-1, 4)) * cur_ratio).astype(np.int32)
            for c, b in zip(cat_ids, bbox):
                if show_all or c == cur_class:
                    cv2.rectangle(_image, (b[0], b[1]), (b[2], b[3]), cats[c]['color'], stroke)
                            
            if select is not None:
                if show_all or cat_ids[select] == cur_class:
                    cv2.rectangle(_image, (bbox[select, 0], bbox[select, 1]), 
                                (bbox[select, 2], bbox[select, 3]), (255, 255, 255, 255), 1+stroke)
                
        if len(tmp_bbox) == 4:
            tmp_bbox = [int(i) for i in tmp_bbox]
            cv2.rectangle(_image, (tmp_bbox[0], tmp_bbox[1]), (tmp_bbox[2], tmp_bbox[3]), (0, 0, 255, 255), 2)

    elif task == 'Object Counting':
        points = g_metadata['annotations'][cur_file_id]['points']
        # print('catId: ', len(g_metadata['annotations'][cur_file_id]['category_ids']))
        # print('InsId: ', len(g_metadata['annotations'][cur_file_id]['instance_ids']))
        # print('points: ', len(points))
        if len(points) >= 1:
            points = (np.reshape(points, (-1, 2)) * cur_ratio).astype(np.int32)
            for c, p in zip(cat_ids, points):
                if show_all or c == cur_class:
                    cv2.circle(_image, (p[0], p[1]), 4+stroke, cats[c]['color'], -1)

            if select is not None:
                if show_all or cat_ids[select] == cur_class:
                    cv2.circle(_image, (points[select, 0], points[select, 1]), 4+stroke, (255, 255, 255, 255), stroke)

    elif task in ['Instance Segmentation', 'Semantic Segmentation', 'Panoptic Segmentation']:
        seg = g_metadata['annotations'][cur_file_id]['seg']
        color_seg = g_metadata['annotations'][cur_file_id]['color_seg'].astype(np.uint8)

        if not show_all and cur_class is not None:
            rr, cc = np.where(seg==ins_ids[cat_ids.index(cur_class)])
            color_seg[rr, cc, :] = [0, 0, 0, 0]

        if select is not None:
            if show_all or cat_ids[select] == cur_class:
                ins_id = ins_ids[select]
                rr, cc = np.where(seg==ins_id)
                color_seg[rr, cc, :] = [255, 255, 255, 255]

        if g_metadata['annotations'][cur_file_id]['color_seg'] is not None:            
            color_seg = cv2.resize(color_seg, (image.shape[1], image.shape[0]))

        alpha = 0.5
        colored_mask = cv2.addWeighted(color_seg, alpha, _image, 1 - alpha, 0)


        _image = np.where(color_seg!=[0,0,0,0], colored_mask, _image)


        if show_bbox:
            for id, cls in zip(g_metadata['annotations'][cur_file_id]['instance_ids'], 
                               g_metadata['annotations'][cur_file_id]['category_ids']):
                
                if show_all or cls == cur_class:
                    tmp = np.zeros(seg.shape)
                    tmp[seg==id] = 1
                    if np.sum(tmp) == 0 or g_metadata['categories'][cls]['type'] == 'sem':
                        continue

                    rr, cc = np.nonzero(tmp)
                    lr = round(np.min(rr)*cur_ratio)
                    lc = round(np.min(cc)*cur_ratio)
                    hr = round(np.max(rr)*cur_ratio)
                    hc = round(np.max(cc)*cur_ratio)
                    cv2.rectangle(_image, (lc, lr),  (hc, hr), (255, 255, 255, 255), 2)

        if mode == 'Polygon':
            node_color = (123, 168, 88, 233)
            if len(tmp_polygon) == 1:
                cv2.circle(_image, tmp_polygon[0], stroke+2, node_color, -1)
            elif len(tmp_polygon) > 1:
                for i in range(len(tmp_polygon) - 1):
                    cv2.circle(_image, tmp_polygon[i], stroke+2, node_color, -1)  # x ,y is the coordinates of the mouse click place
                    cv2.line(_image, pt1=tmp_polygon[i], pt2=tmp_polygon[i + 1], color=(255, 255, 255, 225), thickness=stroke)
                
                cv2.circle(_image, tmp_polygon[-1], stroke+2, node_color, -1)  # x ,y is the coordinates of the mouse click place


    return _image, resized_image

