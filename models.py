from header import *
import torch
import torchvision
from torchvision.io.image import read_image
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset
from PIL import Image

THRESHOLD = 0.5

types =['det', 'count', 'ins', 'sem', 'pan', 'key']
use_model = None
task_type = None
cat_table = {}


class ImageDataset(Dataset):
    def __init__(self, image_paths=None, images=None):
        self.image_paths = image_paths
        self.images = images
        self.size = len(image_paths) if image_paths is not None else len(images)
        self.transform = T.Compose([T.ToTensor()])

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if self.image_paths is not None:
            image = Image.open(self.image_paths[idx]).convert("RGB")
            image_tensor = self.transform(image)
        elif self.images is not None:
            image_tensor = self.transform(self.images[idx])

        return image_tensor


def collate_fn(batch):
    return torch.stack(batch)


def load_models(mode_names: list(), type: str):
    global use_model, task_type, preprocess, cat_table
    task_type = type
    if type == 'det':
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        preprocess = weights.transforms()
        cat_table = weights.meta["categories"]
        use_model = fasterrcnn_resnet50_fpn_v2(weights, box_score_thresh=THRESHOLD)   
        use_model.eval()
    elif task_type == 'count':
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        preprocess = weights.transforms()
        cat_table = weights.meta["categories"]
        use_model = fasterrcnn_resnet50_fpn_v2(weights, box_score_thresh=THRESHOLD)   
        use_model.eval()
    elif task_type == 'ins':
        weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        preprocess = weights.transforms()
        cat_table = weights.meta["categories"]
        use_model = maskrcnn_resnet50_fpn_v2(weights)
        use_model.eval()
    elif task_type == 'sem':
        weights = FCN_ResNet50_Weights.DEFAULT
        preprocess = weights.transforms()
        cat_table = weights.meta["categories"]
        use_model = fcn_resnet50(weights)
        use_model.eval()
    elif task_type == 'pan':
        weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        preprocess = weights.transforms()
        cat_table = weights.meta["categories"]
        use_model = maskrcnn_resnet50_fpn_v2(weights)
        use_model.eval()


def predict(images:list()):
    global use_model, task_type, preprocess
    if use_model is None or task_type is None: return None
    print('predicting...')
    if len(images) > 1:
        dataset = ImageDataset(images=images)
        dataloader = DataLoader(dataset, batch_size=len(dataset), collate_fn=collate_fn)
        input = next(iter(dataloader))

    transform = T.Compose([T.ToTensor()])   
    if task_type == 'det':
        input = [preprocess(transform(images[0]))]
        return predict_det(input)
    elif task_type == 'count':
        input = [preprocess(transform(images[0]))]
        return predict_count(input)
    elif task_type == 'ins':
        input = [preprocess(transform(images[0]))]
        return predict_ins(input)
    elif task_type == 'sem':
        input = preprocess(transform(images[0])).unsqueeze(0)
        return predict_sem(input, input_size=images[0].shape[:2])
    elif task_type == 'pan':
        input = [preprocess(transform(images[0]))]
        return predict_pan(input)
    elif task_type == 'key':
        return predict_key(input)
    else:
        return None
    

def predict_det(input, input_size=None):
    global use_model
    with torch.no_grad():
        r = use_model(input)[0]

    annotations = {}
    annotations['results'] = []
    annotations['cls'] = []
    
    boxes = r['boxes'].cpu().numpy().flatten().tolist()
    labels = r['labels'].cpu().numpy()
    labels = [cat_table[i].lower() for i in labels]
    ins_ids = [i+1 for i, v in enumerate(labels)]
    annotations['cls'].extend([{'name': i, 'type': task_type} for i in labels])
    annotations['results'].append({
        'bboxes': boxes, 
        'points': [],
        'seg': None,
        'category_ids': labels, 
        'instance_ids': ins_ids})

    return annotations


def predict_count(input, input_size=None):
    global use_model
    with torch.no_grad():
        r = use_model(input)[0]

    annotations = {}
    annotations['results'] = []
    annotations['cls'] = []

    boxes = r['boxes'].cpu().numpy()
    points = [((row[0]+row[2])/2, (row[1]+row[3])/2) for row in boxes]
    labels = r['labels'].cpu().numpy()
    labels = [cat_table[i].lower() for i in labels]
    ins_ids = [i+1 for i, v in enumerate(labels)]
    annotations['cls'].extend([{'name': i, 'type': task_type} for i in labels])
    annotations['results'].append({
        'bboxes': [], 
        'points': points,
        'seg': None,
        'category_ids': labels, 
        'instance_ids': ins_ids})
    
    return annotations


def predict_sem(input, input_size=None):
    global use_model
    with torch.no_grad():
        result = use_model(input)['out']

    normalized_masks = result.softmax(dim=1)
    mask = normalized_masks.argmax(1).squeeze().detach().cpu().numpy()

    annotations = {}
    annotations['results'] = []
    annotations['cls'] = []

    labels = sorted(np.unique(mask).astype(np.int32).tolist())
    ins_ids = [-i-1 for i in labels]
    for i, label_id in enumerate(labels):
        mask[mask==label_id] = ins_ids[i]

    mask = cv2.resize(mask, (input_size[1], input_size[0]), interpolation=cv2.INTER_NEAREST)
    cat_ids = [cat_table[i].lower() for i in labels]
    annotations['cls'].extend([{'name': i, 'type': task_type} for i in cat_ids])
    annotations['results'].append({
        'bboxes': [], 
        'points': [],
        'seg': mask.astype(np.int32),
        'category_ids': cat_ids, 
        'instance_ids': ins_ids})

    return annotations


def predict_ins(input, input_size=None):
    global use_model
    with torch.no_grad():
        r = use_model(input)[0]
    boxes = r['boxes'].cpu().numpy()
    labels = r['labels'].cpu().numpy().tolist()
    scores = r['scores'].cpu().numpy().tolist()
    mask = r['masks']
    mask = torch.argmax(mask.squeeze(), dim=0).detach().cpu().numpy()

    annotations = {}
    annotations['results'] = []
    annotations['cls'] = []

    ins_ids = [-i-1 for i, v in enumerate(labels)]
    cat_ids = [cat_table[i].lower() for i in labels]
    annotations['cls'].extend([{'name': i, 'type': task_type} for i in cat_ids])
    for i, ins_id in enumerate(ins_ids):
        if scores[i] >= THRESHOLD:
            mask[mask==i] = ins_id
        else:
            mask[mask==i] = 0

    annotations['results'].append({
        'bboxes': [], 
        'points': [],
        'seg': mask.astype(np.int32),
        'category_ids':cat_ids, 
        'instance_ids': ins_ids})
    
    return annotations


def predict_pan(input, input_size=None):
    global use_model
    with torch.no_grad():
        r = use_model(input)[0]
    boxes = r['boxes'].cpu().numpy()
    labels = r['labels'].cpu().numpy().tolist()
    scores = r['scores'].cpu().numpy().tolist()
    mask = r['masks']
    mask = torch.argmax(mask.squeeze(), dim=0).detach().cpu().numpy()

    annotations = {}
    annotations['results'] = []
    annotations['cls'] = []

    ins_ids = [-i-1 for i, v in enumerate(labels)]
    cat_ids = [cat_table[i].lower() for i in labels]
    annotations['cls'].extend([{'name': i, 'type': task_type if task_type != 'pan' else 'ins'} for i in cat_ids])
    for i, ins_id in enumerate(ins_ids):
        if scores[i] >= THRESHOLD:
            mask[mask==i] = ins_id
        else:
            mask[mask==i] = 0

    annotations['results'].append({
        'bboxes': [], 
        'points': [],
        'seg': mask.astype(np.int32),
        'category_ids':cat_ids, 
        'instance_ids': ins_ids})
    
    return annotations


def predict_key(images:list()):
    return None
