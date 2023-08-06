'''
This is a GUI script powered by DearPyGui.

Author: Lucas Chu
Date: July 24, 2023
'''
from header import *
import utils
from helper import *
import kernel
import dearpygui.dearpygui as dpg
import webbrowser
import array, string, time
import gc

# Image area size
EDITOR_WIDTH = 1000
EDITOR_HEIGHT = 750

# Window size
WINDOW_WIDTH = EDITOR_WIDTH+330
WINDOW_HEIGHT = EDITOR_HEIGHT+100

ICON_HEIGHT = 100
SIDE_MENU_WIDTH = 300
SIDE_MENU_HEIGHT = EDITOR_HEIGHT

BTN_HEIGHT = 30

# Menu bar height and sub-window interval
TOP_PADDING = 20
WINDOW_INTERVAL = 15

ZOOM_MAX = 1.8
ZOOM_MIN = 0.6
ZOOM_STEP = 0.4

# Global variables
refresh = False

class_list = []
source_files = []
project_path = ''
mouse_drag_flag = False
font_scale = 1.0
show_all = True
freeze_click = False
next_freeze_click = False
cur_move_xy , cur_down_xy, cur_drag_xy= [0 ,0], [0, 0] ,[1, 1]
supported_image_formats = ['jpg', 'jpeg', 'png', 'bmp']
stroke = 2

foreground_flag = True
show_bbox = False
ai_for_all = False

workspace = [0, 0, 0, 0, 0, 0] # x y w h offset_x, offset_y

annotation_task_list = [' ', 'Object Counting', 'Object Detection', 'Instance Segmentation', 'Semantic Segmentation', 'Panoptic Segmentation']
annotation_ratio_btn_list = ['annotatino_mode_none', 'annotation_mode_counting', 'annotation_mode_detection', 'annotation_mode_instance', 'annotation_mode_semantic', 'annotation_mode_panoptic']
segmentation_op_list = ['Select', 'Superpixel', 'Polygon']
segmentation_nosuper_op_list = ['Select', 'Polygon']

# File paths    
file_open_path = os.path.join(os.getcwd(), 'inputs')
file_save_path = os.path.join(os.getcwd(), 'outputs')

# Helper functions
def dpg_image(_image, width=EDITOR_WIDTH, height=EDITOR_HEIGHT):
    template_image = np.zeros((height, width, 4), dtype=np.uint8)
    template_image[:_image.shape[0], :_image.shape[1], :] = _image
    return template_image.flatten().astype(float) / 255.0

    
# Initialize image
original_image = np.zeros((1, 1, 4), dtype=np.uint8)
image = np.zeros((1, 1, 4), dtype=np.uint8)
image_dpg = dpg_image(image)

dpg.create_context()


def cvt_coord_to_image(coord):
    global workspace
    return [coord[0]-workspace[0]+workspace[4], coord[1] - workspace[1] + workspace[5]]


def read_image(filename):
    global project_path
    try:
        _image = cv2.imread(os.path.join(project_path, filename))
        if _image is not None:
            _image = cv2.cvtColor(_image, cv2.COLOR_BGR2RGB)
        return _image   
    except:
        print('File not found!')
        return None
 

def load_image2window(_image):
    global original_image, image, image_dpg
    ratio = 1.0
    if _image is not None:
        # Generate top paddings for menu bar
        # rows_with_zeros = np.zeros((TOP_PADDING, _image.shape[1], 3), dtype=np.uint8)
        # _image = np.concatenate((rows_with_zeros, _image), axis=0)
        ratio = min(EDITOR_WIDTH / _image.shape[1], EDITOR_HEIGHT / _image.shape[0])
        ratio = round(ratio-0.02, 2)
        _image = cv2.resize(_image, (round(_image.shape[1]*ratio), round(_image.shape[0]*ratio)))

        alpha_channel = np.full((_image.shape[0], _image.shape[1]), 255, dtype=np.uint8)
        image_rgba = np.dstack((_image, alpha_channel))
        # image_rgba[:TOP_PADDING, :, 3] = 0    
        original_image = image_rgba
        image = original_image.copy()
    else:
        image = None
    return image, ratio


def is_mouse_in_image_panel():
    global cur_move_xy, workspace
    if cur_move_xy[0] < workspace[0]+workspace[2] and cur_move_xy[0] >= workspace[0] and \
        cur_move_xy[1] >= workspace[1] and cur_move_xy[1] <= workspace[1]+workspace[3]:
        return True
    else:
        return False
    

def is_class_selected():
    if kernel.cur_class is None:
        return False
    else:
        return True
    

def is_in_foreground():
    global foreground_flag
    return foreground_flag


def force_refresh():
    global refresh
    refresh = True


def clean_tmp():
    kernel.tmp_polygon = []
    kernel.tmp_bbox = []
    kernel.select = None
    # force_refresh()


def is_mouse_effective():
    condition_1 = is_mouse_in_image_panel()
    condition_2 = is_class_selected()
    condition_3 = is_in_foreground()
    
    return condition_1 and condition_2 and condition_3

# Menu bar area
def is_menu_area(x, y):
    if x > 0 and x < 300 and y > 0 and y < 30:
        return True
    else:
        return False
    

def freeze_mouse_click(x, y):
    global freeze_click, next_freeze_click
    
    freeze_click = next_freeze_click
    if is_menu_area(x, y):
        next_freeze_click = True
    else:
        next_freeze_click = False

    return freeze_click

    
def config_class_list():
    if len(kernel.g_metadata['categories']) > 0:
        if kernel.cur_class not in kernel.g_metadata['categories']:
            kernel.cur_class = list(kernel.g_metadata['categories'].keys())[0]

        class_list = [v['name'] for i, v in kernel.g_metadata['categories'].items()]
        default_class = kernel.g_metadata['categories'][kernel.cur_class]['name']
    else:
        kernel.cur_class = None
        class_list = []
        default_class = ''

    class_list = sorted(class_list)
    dpg.configure_item("class_show_dropdown", items=class_list, default_value=default_class)
    dpg.configure_item("class_assign_dropdown", items=class_list, default_value=default_class)



def mouse_down_callback(sender, mouse_data):
    global cur_down_xy, cur_drag_xy, cur_move_xy, mouse_drag_flag, click_start_xy, click_end_xy
    if not is_mouse_effective(): return
    if not mouse_drag_flag:
        click_start_xy = cur_move_xy
        cur_down_xy = cur_move_xy
        cur_drag_xy = [0, 0]
        mouse_drag_flag = True
        force_refresh()


def mouse_release_callback(sender, mouse_data):
    global cur_drag_xy, cur_down_xy, cur_move_xy, mouse_drag_flag, click_start_xy, click_end_xy
    if mouse_drag_flag:
        mouse_drag_flag = False

    if not is_mouse_effective(): return

    # Click callback
    # if _drag_flag:
    #     # mouse_drag_flag = False
    #     click_end_xy = cur_move_xy
    #     if click_end_xy[0] == click_start_xy[0] and click_end_xy[1] == click_start_xy[1]:
    #         click_xy = cvt_coord_to_image(click_end_xy)
    #         if kernel.task == 'Object Counting' and kernel.mode == 'Click':
    #             kernel.on_counting_click(click_xy[0], click_xy[1])
    #         elif kernel.task in ['Instance Segmentation', 'Semantic Segmentation', 'Panoptic Segmentation'] and kernel.mode == 'Superpixel':
    #             if kernel.pre_mask is None:
    #                 show_loading('preprocess', [], 'Loading...')

    #             kernel.on_segmentation_click(click_xy[0], click_xy[1])
    #         elif kernel.task in ['Instance Segmentation', 'Semantic Segmentation', 'Panoptic Segmentation'] and kernel.mode == 'Polygon':
    #             kernel.on_polygon(click_xy[0], click_xy[1])
    #         elif kernel.mode == 'Select':
    #             kernel.on_select(click_xy[0], click_xy[1])
    
    # drag callback
    if kernel.task == 'Object Detection' and kernel.mode == 'Rectangle':
        kernel.on_rectangle_end(cvt_coord_to_image(cur_down_xy), cvt_coord_to_image(cur_drag_xy))


    force_refresh()


def mouse_move_callback(sender, mouse_data):
    global cur_move_xy
    cur_move_xy = mouse_data


def mouse_drag_callback(sender, mouse_data):
    global cur_drag_xy, cur_down_xy, workspace, image

    cur_drag_xy = [cur_down_xy[0]+mouse_data[1], cur_down_xy[1]+mouse_data[2]]
    drag_xy = cvt_coord_to_image(cur_drag_xy)
    down_xy = cvt_coord_to_image(cur_down_xy)

    if not is_mouse_effective(): 
        clean_tmp() 
        return
        # if kernel.task in ['Object Detection'] and kernel.mode == 'Rectangle':
        #     kernel.on_rectangle_end(down_xy, drag_xy)

        # return

    if kernel.task in ['Object Detection'] and kernel.mode == 'Rectangle':
        if mouse_drag_flag and np.array(drag_xy).any() > 0:
            kernel.on_rectangle(down_xy, drag_xy)

    elif kernel.task in ['Instance Segmentation', 'Semantic Segmentation', 'Panoptic Segmentation'] and kernel.mode == 'Superpixel':
        kernel.on_draw(drag_xy[0], drag_xy[1])

    elif kernel.mode == 'Select':
        if workspace[4] - mouse_data[1] > 0 and workspace[2]+workspace[4]-mouse_data[1] < image.shape[1]:
            workspace[4] -= mouse_data[1]
        if workspace[5] - mouse_data[2] > 0 and workspace[3]+workspace[5]-mouse_data[2] < image.shape[0]:
            workspace[5] -= mouse_data[2]

    force_refresh()


def mouse_click_callback(sender, mouse_data):
    global cur_move_xy, mouse_drag_flag, click_start_xy, cur_drag_xy, cur_down_xy
    # print('click')
    if freeze_mouse_click(cur_move_xy[0], cur_move_xy[1]): return
    if not is_mouse_effective(): return
    
    if mouse_drag_flag:
        # mouse_drag_flag = False
        click_xy = cvt_coord_to_image(click_start_xy)
        if kernel.task == 'Object Counting' and kernel.mode == 'Click':
            kernel.on_counting_click(click_xy[0], click_xy[1])
        elif kernel.task in ['Instance Segmentation', 'Semantic Segmentation', 'Panoptic Segmentation'] and kernel.mode == 'Superpixel':
            if kernel.pre_mask is None:
                show_loading('preprocess', [], 'Loading...')

            kernel.on_segmentation_click(click_xy[0], click_xy[1])
        elif kernel.task in ['Instance Segmentation', 'Semantic Segmentation', 'Panoptic Segmentation'] and kernel.mode == 'Polygon':
            kernel.on_polygon(click_xy[0], click_xy[1])
        elif kernel.mode == 'Select':
            r = kernel.on_select(click_xy[0], click_xy[1])
            if r is not None:
                _id = kernel.g_metadata['annotations'][kernel.cur_file_id]['category_ids'][r]
                dpg.set_value('image_hover_info', kernel.g_metadata['categories'][_id]['name'])

    
def keyboard_release_callback(sender, key_data):
    global image
    # print('key: ', sender, key_data)
    if key_data == 265: # up
        zoom_callback('Zoom In', None)
    elif key_data == 264: # down
        zoom_callback('Zoom Out', None)
    elif key_data == 263: # left
        file_move_callback('file_move_left', None)
    elif key_data == 262: # right
        file_move_callback('file_move_right', None)
    elif key_data == 256: # esc:
        if kernel.mode == 'Polygon':
            polygon_finish_callback('polygon_cancel', None)
        elif kernel.mode == 'Select':
            kernel.select = None
    elif key_data == 32 or key_data == 257: # space or enter
        if kernel.mode == 'Polygon':
            polygon_finish_callback('polygon_finish', None)
    elif key_data == 259: # backspace, delete
        if kernel.mode == 'Select' and kernel.select is not None:
            annotation_delete_callback('', None)

    move_step = 200
    if kernel.mode == 'Select':
        if key_data == 259 and kernel.select is not None: # backspace, delete
            annotation_delete_callback('', None)
        elif key_data == 65: # a
            if workspace[4] - move_step > 0:
                workspace[4] -= move_step
            else:
                workspace[4] = 0

        elif key_data == 68: # d
            if workspace[2]+workspace[4]+move_step < image.shape[1]:
                workspace[4] += move_step
            else:
                workspace[4] = image.shape[1] - workspace[2]
        elif key_data == 87: # w
            if workspace[5] - move_step > 0:
                workspace[5] -= move_step
            else:
                workspace[4] = 0

        elif key_data == 83: # s
            if workspace[3]+workspace[5]+move_step < image.shape[0]:
                workspace[5] += move_step
            else:
                workspace[5] = image.shape[0] - workspace[3]

    force_refresh()


# add a mouse&keyboard handler registry
with dpg.handler_registry():
    dpg.add_mouse_down_handler(callback=mouse_down_callback)
    dpg.add_mouse_move_handler(callback=mouse_move_callback)
    dpg.add_mouse_drag_handler(callback=mouse_drag_callback)
    dpg.add_mouse_release_handler(callback=mouse_release_callback)
    dpg.add_mouse_click_handler(callback=mouse_click_callback)
    dpg.add_key_release_handler(callback=keyboard_release_callback)


# add a font registry
with dpg.font_registry():
    # first argument ids the path to the .ttf or .otf file
    regular_font = dpg.add_font("resources/Roboto-Regular.ttf", 20)
    small_font = dpg.add_font("resources/Roboto-Regular.ttf", 16)
    bold_font = dpg.add_font("resources/Roboto-Bold.ttf", 20)
    italic_font = dpg.add_font("resources/Roboto-Italic.ttf", 17)
    button_font = dpg.add_font("resources/Roboto-Bold.ttf", 19)
    default_font = regular_font


with dpg.texture_registry(show=False):
    dpg.add_dynamic_texture(width=EDITOR_WIDTH, height=EDITOR_HEIGHT, default_value=image_dpg, tag="source_image")
    width, height, channels, icon = dpg.load_image("resources/icon.png")
    dpg.add_static_texture(width=width, height=height, 
                           default_value=icon, tag="icon_image")

 
def create_project(_path, _metadata=None):
    global project_path, source_files
    project_path = _path
    path_max = 35
    show_path = ''
    for i in range(0, len(_path), path_max):
        if i+path_max >= len(_path):
            show_path += _path[i:]
            break
        else:
            show_path += _path[i:i+path_max] + '\n'

    dpg.set_value("Project Path", show_path)
    dpg.show_item("Project Path")
    result_list = []
    for i in supported_image_formats:
        format = '*.' + i
        result_list.extend(glob.glob(os.path.join(project_path, format)))
            
    source_files = sorted([item.split('/')[-1] for item in result_list])
    if len(source_files) > 0:
        kernel.init_workspace(project_path, source_files, _metadata)
        dpg.configure_item("file_dropdown", items=source_files, default_value=source_files[0])
        dpg.configure_item("task_dropdown", items=annotation_task_list, default_value=kernel.task)
        switch_task(kernel.task)
        config_class_list()
        set_cur_file(source_files[0])
        config_side_menu()
        kernel.mode = 'Select'



def import_project(_path, _metadata, default_type='picA'):
    if default_type == 'picA':
        if _metadata['info']['task_type'] == kernel.g_metadata['info']['task_type']:
            kernel.g_metadata['categories'].update(_metadata['categories'])
            for key, value in _metadata['annotations'].items():
                if len(value['category_ids']) > 0:
                    kernel.load_annotations(key, value, _path)
        
    elif default_type == 'COCO':
        pass

    
    config_class_list()
    force_refresh()


# Create, import, and export a project callback
def okay_callback(sender, data):
    global source_files, supported_image_formats, foreground_flag
    _path = data['file_path_name']
    def dialog_option(sender, data):
        global foreground_flag
        dpg.delete_item('dialog')
        foreground_flag = True

    if sender == "folder_open_dialog":
        if os.path.exists(os.path.join(_path, 'annotations/annotations.json')):
            show_info("Warning", "This folder already contains a project. You can import it.", dialog_option, sender)
        else:
            create_project(_path)
            foreground_flag = True

    elif sender == "folder_import_dialog":
        import_metadata = utils.parse_annotation_json(os.path.join(data['current_path'], 'annotations/annotations.json'))
        if import_metadata is not None:
            if kernel.g_metadata is None:
                create_project(_path, import_metadata)
            else:
                import_project(_path, import_metadata)
            
            foreground_flag = True
        else:
            show_info("Warning", 
                      "This is not picA's project or this folder does not contain a valid annotation file. \n You should import a project-level folder, not a json file.", 
                      dialog_option, 
                      sender)
    elif sender == "folder_export_dialog":
        show_loading('export_2_coco', [_path], 'Exporting...')
    

def cancel_callback(sender, data):
    # print("Cancel is clicked. Sender: ", sender)
    pass

def undo_callback(sender, data):
    pass

def create_class_callback(sender, data):
    new_class = dpg.get_value("Create New Class")
    dpg.set_value("Create New Class", "")
    new_class = utils.filter_input(new_class)
    if len(new_class) <= 0: return

    new_class = new_class[:24] if len(new_class) > 24 else new_class
    if new_class.lower() not in [i['name'].lower() for i in kernel.g_metadata['categories'].values()]:
        kernel.new_class(new_class)
        config_class_list()
        update_instance_listbox(kernel.get_class_id(new_class))
        clean_tmp()


def update_instance_listbox(_class_id):
    if kernel.task in ['Instance Segmentation', 'Panoptic Segmentation']:
        dpg.show_item('show_instance_bbox')
        if _class_id is None:
            dpg.hide_item('instance_selection_header')
            
        elif kernel.is_instance(_class_id):
            indices = [i for i, v in enumerate(kernel.g_metadata['annotations'][kernel.cur_file_id]['category_ids']) \
                       if v == _class_id]
            ins_ids = [kernel.g_metadata['annotations'][kernel.cur_file_id]['instance_ids'][i] for i in indices]
            dpg.show_item('instance_selection_header')
            dpg.set_value("instance_class_label", '' + kernel.g_metadata['categories'][_class_id]['name'])
            if type(ins_ids) is not list:
                ins_ids = [ins_ids]
            
            dpg.configure_item("instance_listbox", items=[i+1 for i, v in enumerate(ins_ids)])
            # dpg.set_value('instance_id_label', value=len(ins_ids))
        else:
            dpg.hide_item('instance_selection_header')
    
    else:
        dpg.hide_item('show_instance_bbox')


def class_show_callback(sender, data):
    kernel.cur_class = kernel.get_class_id(data)
    kernel.new_instance(kernel.cur_class)
    update_instance_listbox(kernel.cur_class)


def class_assign_callback(sender, data):
    if kernel.on_assign_class(data):
        dpg.set_value('image_hover_info', data)
        kernel.select = None
        

def delete_class_callback(sender, data):
    if len(kernel.g_metadata['categories']) <= 0: return
    show_info("Warning", "This category "+kernel.g_metadata['categories'][kernel.cur_class]['name']+" and related annotations will be deleted over all images.\nThis operation is irreversible. Do you really want to DELETE? ", proceed_class_delete, sender)
    clean_tmp()


def proceed_class_delete(sender, data):
    global foreground_flag
    print('delete class ')
    if data == 'okay':
        kernel.delete_class_global(kernel.cur_class)
        config_class_list()
        update_instance_listbox(None)
        force_refresh()
    elif data == 'cancel':
        pass
        
    dpg.delete_item('dialog')
    foreground_flag = True


def show_all_class_callback(sender, data):
    global show_all
    show_all = data
    print('Show all was clicked. ', data)



def focus_on_file_callback(sender, data):
    set_cur_file(data)


def set_cur_file(_filename):
    global workspace, source_files, project_path
    dpg.configure_item("file_dropdown", items=source_files, default_value=_filename)
    clean_tmp()
    show_loading('on_file_start', [_filename, project_path], 'Loading...')
    force_refresh()


def create_instance_callback(sender, data): 
    if kernel.new_instance(kernel.cur_class):
        print('create instance')
        update_instance_listbox(kernel.cur_class)


def file_move_callback(sender, data):
    global source_files
    if kernel.cur_file_id is None: return

    source_file_index = source_files.index(kernel.cur_file_id)
    if sender == 'file_move_left' and source_file_index - 1 >= 0:
        source_file_index -= 1
        set_cur_file(source_files[source_file_index])
    elif sender == 'file_move_right' and source_file_index + 1 < len(source_files):
        source_file_index += 1
        set_cur_file(source_files[source_file_index])


def font_callback(sender, data):
    global font_scale
    if sender == "Increase font size +":
        font_scale = min(1.4, font_scale + 0.2)
    elif sender == "Decrease font size":
        font_scale = max(0.6, font_scale - 0.2)
    
    dpg.set_global_font_scale(font_scale)


def dialog_option(sender, data):
    global foreground_flag
    if sender == 'task_dropdown':
        if data == 'okay':
            dpg.stop_dearpygui()
        elif data == 'cancel':
            dpg.set_value(sender, kernel.task)
            dpg.delete_item('dialog')
    
    foreground_flag = True
    

def config_bottom_op():
    global annotation_ratio_btn_list, segmentation_op_list, segmentation_nosuper_op_list
    if kernel.task in ['Instance Segmentation', 'Semantic Segmentation', 'Panoptic Segmentation']:
        for i in annotation_ratio_btn_list[3:]:
            if kernel.use_superpixel:
                dpg.configure_item(i, items=segmentation_op_list)
            else:
                if kernel.mode == 'Polygon':
                    default_value = 'Polygon'
                else:
                    default_value = 'Select'
                    kernel.mode = default_value

                dpg.configure_item(i, items=segmentation_nosuper_op_list, default_value=default_value)
                annotation_mode_callback('', default_value)



def show_loading(_task, args, msg):
    global image, foreground_flag, original_image, ai_for_all
    global segmentation_nosuper_op_list
    global segmentation_op_list
    global source_files, project_path
    dpg.configure_item("LoadingWindow", show=True)
    dpg.set_value("Loading Message", msg)
    foreground_flag = False
    if _task == 'preprocess':
        kernel.preprocess(image[:, :, :3])
    elif _task == 'on_file_start':
        filename = args[0]
        kernel.on_file_end()
        _image = read_image(filename)
        original_image_shape = _image.shape
        _image, ratio = load_image2window(_image)
        kernel.on_file_start(filename, _image, [ratio, original_image_shape])
        if _image is not None:
            workspace[2] = _image.shape[1]
            workspace[3] = _image.shape[0]
            workspace[4] = 0
            workspace[5] = 0

        config_bottom_op()
        update_instance_listbox(kernel.cur_class)


    elif _task == 'save_workspace':
        if kernel.g_metadata is not None:
            kernel.save_workspace()
            dpg.set_value("Last Save", "Last Save: " + time.strftime("%m-%d %H:%M", time.localtime()))
            dpg.show_item("Last Save")

    elif _task == 'export_2_coco':
        if kernel.g_metadata is not None:
            result = export_2_coco_format(kernel.g_metadata, args[0])

    elif _task == 'load_model':
        kernel.load_models()

    elif _task == 'ai_annotate_btn':
        if ai_for_all:
            input = [os.path.join(project_path, i) for i in source_files]
        else:
            input = original_image[:, :, :3]
        if kernel.predict(input):
            print('AI annotate works!')
            config_class_list()
            force_refresh(  )
        
    # Hide the loading indicator when the task is completed
    dpg.configure_item("LoadingWindow", show=False)
    foreground_flag = True


def task_select_callback(sender, data):
    if kernel.annotation_start:
        show_info("Warning", "You are about to change the task. All the unsaved data will be lost.", dialog_option, sender)
    else:
        switch_task(data)


def make_instance_callback(sender, data):
    if not kernel.is_instance(kernel.cur_class):
        kernel.semantic2instance(kernel.cur_class)
        kernel.new_instance(kernel.cur_class)
        update_instance_listbox(kernel.cur_class)
        # dpg.configure_item("make_instance", label="Make it Class")


def annotation_mode_callback(sender, data):
    kernel.mode = data
    if data == 'Polygon':
        dpg.show_item('polygon_finish')
        dpg.show_item('polygon_cancel')
        dpg.show_item('annotation_delete')
        dpg.hide_item('Class Assign')
    elif data == 'Select':
        dpg.hide_item('polygon_finish')
        dpg.show_item('polygon_cancel')
        dpg.show_item('annotation_delete') 
        dpg.show_item('Class Assign')
    else:
        dpg.hide_item('polygon_finish')
        dpg.hide_item('polygon_cancel')
        dpg.hide_item('annotation_delete')
        dpg.hide_item('Class Assign')
    
    # print('annotation mode: ', data)
    clean_tmp()
    force_refresh()


def polygon_finish_callback(sender, data): 
    if sender == 'polygon_finish':
        if len(kernel.tmp_polygon) <= 0: return
        kernel.on_polygon_end()
    elif sender == 'polygon_cancel':
        clean_tmp()
    
    force_refresh()


def annotation_delete_callback(sender, data):
    if kernel.task == 'Object Counting':
        kernel.on_delete_selected()
    elif kernel.task == 'Object Detection':
        kernel.on_delete_selected()
    elif kernel.task in ['Instance Segmentation', 'Semantic Segmentation', 'Panoptic Segmentation']:
        if kernel.mode == 'Polygon':
            kernel.on_polygon_delete()
        elif kernel.mode == 'Select':
            kernel.on_delete_selected()
    
    force_refresh()


def menu_drop_callback(sender, data):
    print("Menu item: ", sender, "was dropped on: ", data)




def zoom_callback(sender, data):
    global image, workspace
    if image.shape[0] > 1 and image.shape[1] > 1:
        if sender == "Zoom In":
            kernel.cur_ratio = kernel.cur_ratio + ZOOM_STEP if kernel.cur_ratio + ZOOM_STEP < ZOOM_MAX else ZOOM_MAX
            
        elif sender == "Zoom Out":
            kernel.cur_ratio = kernel.cur_ratio - ZOOM_STEP if kernel.cur_ratio - ZOOM_STEP > ZOOM_MIN else ZOOM_MIN
        elif sender == "Actual Size":
            kernel.cur_ratio = 1.0

    workspace[4] = 0
    workspace[5] = 0
    clean_tmp()
    force_refresh()

# def filter_list(sender, data):
#     global source_files
#     filter_text = dpg.get_value("file_filter")
#     filtered_files = [item for item in source_files if filter_text in item]
#     if len(filtered_files) > 0:
#         selected_filename = filtered_files[0]
#         set_cur_file(selected_filename)
        
def save():
    global original_image
    kernel.on_file_end()
    show_loading('save_workspace', [], 'Saving...')


def quit():
    save()
    gc.collect()
    print('quit!')


### Menu Setup
# Dialog Window

def show_info(title, message, dialog_callback, launch_source):
    global foreground_flag
    foreground_flag = False
    with dpg.mutex():
        viewport_width = dpg.get_viewport_client_width()
        viewport_height = dpg.get_viewport_client_height()
        
        with dpg.window(label=title, modal=True, no_close=False, no_bring_to_front_on_focus=False, tag='dialog', on_close=lambda s, d: dialog_callback(launch_source, 'cancel'), no_move=True, 
                        pos=[460, 360]) as item:
            dpg.add_text(message)
            with dpg.group(horizontal=True, width=0):
                dpg.add_button(label="OK", width=75, callback=lambda s, d: dialog_callback(launch_source, 'okay'))
                dpg.bind_item_theme(dpg.last_item(), 'okay_button_theme')
                dpg.bind_item_font(dpg.last_item(), button_font)
                dpg.add_spacer(width=20)
                dpg.add_button(label="Cancel", width=75, callback=lambda s, d: dialog_callback(launch_source, 'cancel'))
                dpg.bind_item_theme(dpg.last_item(), 'cancel_button_theme')
                dpg.bind_item_font(dpg.last_item(), button_font)
    # guarantee these commands happen in another frame
    dpg.split_frame()
    # width = dpg.get_item_width(item)
    # height = dpg.get_item_height(item)
    # dpg.set_item_pos(item, [viewport_width // 2 - width // 2, viewport_height // 2 - height // 2])


def config_side_menu():

    if kernel.task in ['Object Counting', 'Object Detection', 'Instance Segmentation', 'Semantic Segmentation', 'Panoptic Segmentation']:
        dpg.show_item('class_selection_header')
    else:
        dpg.hide_item('class_selection_header')


    if kernel.task == 'Panoptic Segmentation':
        dpg.show_item('make_instance')
    else:
        dpg.hide_item('make_instance')

    if kernel.task in ['Instance Segmentation', 'Panoptic Segmentation']:
        dpg.show_item('show_instance_bbox')
    else:
        dpg.hide_item('show_instance_bbox')


def switch_task(selected_task):
    global annotation_task_list, annotation_ratio_btn_list

    kernel.task = selected_task
    for i, v in enumerate(annotation_task_list):
        if v == selected_task:
            dpg.show_item(annotation_ratio_btn_list[i])
        else:
            dpg.hide_item(annotation_ratio_btn_list[i])

    if selected_task in ['Object Counting', 'Object Detection', 'Instance Segmentation', 'Semantic Segmentation', 'Panoptic Segmentation']:
        dpg.show_item('Stroke Slider Selection')
    else:
        dpg.hide_item('Stroke Slider Selection')



def show_ins_bbox_callback(sender, data):
    global show_bbox
    show_bbox = data
    force_refresh()


def ai_annotate_callback(sender, data):
    global foreground_flag, ai_for_all
    def dialog_return_callback(_sender, _data):
        global foreground_flag
        dpg.delete_item('dialog')
        time.sleep(0.05)
        if _data == 'okay':
            show_loading(sender, [], 'Annotating...')
        elif _data == 'cancel':
            foreground_flag = True

    show_info("Warning", "You are using AI model to annotate the image and \nexisting data for this file will be lost.", dialog_return_callback, '')
    

def ai_annotate_all_callback(sender, data):
    global ai_for_all
    ai_for_all = data


def stroke_callback(sender, data):
    global stroke
    stroke = data
    force_refresh()


def menu_smart_callback(sender, data):
    if sender == 'Check Superpixel':
        kernel.use_superpixel = data
        config_bottom_op()

    elif sender == 'Check AI model':
        kernel.use_ai_model = data
        if kernel.use_ai_model:
            show_loading('load_model', [], 'Loading...')
            dpg.show_item('ai_annotate_btn')
        else:
            dpg.hide_item('ai_annotate_btn')


def menu_create_project_callback(sender, data):
    global foreground_flag
    def dialog_return_callback(_sender, _data):
        global foreground_flag
        if _sender == 'Second Create':
            if _data == 'okay':
                dpg.stop_dearpygui()
            elif _data == 'cancel':
                foreground_flag = True
        
        elif _sender == 'First Create':
            foreground_flag = True
    
        dpg.delete_item('dialog')

    if kernel.g_metadata is not None:
        show_info("Warning", "You need to quit and create another project. All the unsaved data will be lost.", dialog_return_callback, 'Second Create')
    else:
        if kernel.task in ['Object Counting', 'Object Detection', 'Instance Segmentation', 'Semantic Segmentation', 'Panoptic Segmentation']:
            dpg.show_item("folder_open_dialog")
            foreground_flag = False
        else:
            show_info("Warning", "Please select a task !", dialog_return_callback, 'First Create')



def menu_export_project_callback(sender, data):
    global foreground_flag
    if sender == 'Export as COCO' and kernel.g_metadata is not None:
        dpg.show_item("folder_export_dialog")
        foreground_flag = False

    pass


def menu_import_project_callback(sender, data):
    global foreground_flag
    dpg.show_item("folder_import_dialog")
    foreground_flag = False
    

def menu_reload_images_callback(sender, data):
    global project_path, source_files
    if kernel.g_metadata is not None:
        result_list = []
        for i in supported_image_formats:
            format = '*.' + i
            result_list.extend(glob.glob(os.path.join(project_path, format)))
                    
        target_files = sorted([item.split('/')[-1] for item in result_list])
        for item in target_files:
            if item not in source_files:
                source_files.append(item)
                kernel.new_metadata_annotation(item)

        dpg.configure_item("file_dropdown", items=source_files)



def register_theme_fonts():
    dpg.bind_font(default_font)
    with dpg.theme(tag="hyperlinkTheme"):
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, [0, 0, 0, 0])
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, [0, 0, 0, 0])
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, [29, 151, 236, 25])
            dpg.add_theme_color(dpg.mvThemeCol_Text, [29, 151, 236])
            
    with dpg.theme(tag="okay_button_theme"):
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, [80, 176, 240])
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, [70, 120, 230])
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, [90, 196, 255])
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 10)
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 5, 5)


    with dpg.theme(tag="cancel_button_theme"):
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, [150, 150, 150])
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, [100, 100, 100])
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, [180, 180, 180])
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 10)
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 5, 5)


    with dpg.theme(tag="delete_button_theme"):
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, [244, 122, 24])
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, [201, 97, 15])
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, [255, 150, 37])
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 10)
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 5, 5)

    with dpg.theme(tag="magic_button_theme"):
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, [53, 165, 60])
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, [20, 115, 35])
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, [75, 198, 86])
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 10)
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 5, 5)
   


def register_dialog():
    dpg.add_file_dialog(
        directory_selector=True, show=False, callback=okay_callback, tag="folder_open_dialog",
        cancel_callback=cancel_callback, width=600 ,height=400)

    dpg.add_file_dialog(
        directory_selector=True, show=False, callback=okay_callback, tag="folder_import_dialog",
        cancel_callback=cancel_callback, width=700 ,height=400)
    

    dpg.add_file_dialog(
        directory_selector=True, show=False, callback=okay_callback, tag="folder_export_dialog",
        cancel_callback=cancel_callback, width=700 ,height=400)


register_dialog()

# Main Window
with dpg.window(label="Image", tag='Image', width=WINDOW_WIDTH, height=WINDOW_HEIGHT,
                 menubar=True, autosize=False, no_close=True, no_scrollbar=True, 
                 no_collapse=True, no_resize=True):
    
    register_theme_fonts()
    with dpg.menu_bar(tag="Main Menu", parent="Image"):
        with dpg.menu(label="File", drop_callback=menu_drop_callback, tracked=True, tag="File"):
            dpg.add_menu_item(label="New Project", callback=menu_create_project_callback)
            dpg.add_menu_item(label="Import Project", callback=menu_import_project_callback)
            dpg.add_menu_item(label="Reload Images", callback=menu_reload_images_callback)
            dpg.add_menu_item(label="Save", callback=save)
            dpg.add_menu_item(label="Export as COCO", callback=menu_export_project_callback, tag="Export as COCO", show=False)
            dpg.add_menu_item(label="Exit", callback=lambda: dpg.stop_dearpygui())

        with dpg.menu(label="View"):
            dpg.add_menu_item(label="Actual Size", callback=zoom_callback, tag="Actual Size")
            dpg.add_menu_item(label="Zoom In", callback=zoom_callback, tag="Zoom In")
            dpg.add_menu_item(label="Zoom Out", callback=zoom_callback, tag="Zoom Out")
            dpg.add_menu_item(label="Font Size +", callback=font_callback, tag="Increase font size", show=False)
            dpg.add_menu_item(label="Font Size -", callback=font_callback, tag="Decrease font size", show=False)


        with dpg.menu(label="Smart Annotation"):
            dpg.add_menu_item(label="Superpixel", check=True, default_value=False, 
                              callback=menu_smart_callback, tag="Check Superpixel")
            dpg.add_menu_item(label="AI model", check=True, default_value=False,
                              callback=menu_smart_callback, tag="Check AI model")

        dpg.add_menu_item(label="Undo", callback=undo_callback, tag="Undo")
        with dpg.tooltip("Undo"):
                dpg.add_text("Will be available soon", color=[150, 150, 150])
                dpg.bind_item_font(dpg.last_item(), small_font)

        with dpg.menu(label="Help"):
            dpg.add_text(utils.help_info, color=[200, 200, 200])
            dpg.add_button(label='PicA\'s Website', callback=lambda:webbrowser.open(utils.github_url))
            dpg.bind_item_theme(dpg.last_item(), "hyperlinkTheme")
            dpg.add_separator()
            dpg.add_text(u"Version: " + utils.version, color=[200, 200, 200])

    with dpg.group(horizontal=True, width=0, tag="Main Panel", parent='Image'):
        dpg.add_image("source_image", tracked=False, parent='Main Panel')
        with dpg.tooltip(dpg.last_item(), tag='image_hover_info_dialog', show=False):
            dpg.add_text("", tag="image_hover_info", show=True)
            dpg.bind_item_font(dpg.last_item(), small_font)

        with dpg.child_window(width=SIDE_MENU_WIDTH, height=SIDE_MENU_HEIGHT, pos=[EDITOR_WIDTH+WINDOW_INTERVAL, TOP_PADDING]):
            dpg.add_text("", tag="Last Save", show=False)
            dpg.add_text("Project Path")
            dpg.bind_item_font(dpg.last_item(), bold_font)
            dpg.add_text("", tag="Project Path", show=False, color=[201, 201, 201])
            dpg.bind_item_font(dpg.last_item(), italic_font)
            dpg.add_spacer(height=5)
            dpg.add_separator()
            dpg.add_text("File Selection", tag= "File Selection")
            dpg.bind_item_font(dpg.last_item(), bold_font)
            # dpg.add_input_text(label="Filter ", callback=filter_list, tag="file_filter")
            widget = dpg.add_combo(label="", items=source_files, callback=focus_on_file_callback, tag="file_dropdown")
            
            with dpg.group(horizontal=True, width=0):
                dpg.add_button(arrow=True, direction=dpg.mvDir_Left, user_data=widget, callback=file_move_callback, tag="file_move_left")
                dpg.add_button(arrow=True, direction=dpg.mvDir_Right, user_data=widget, callback=file_move_callback, tag="file_move_right")

            dpg.add_separator()

            with dpg.collapsing_header(label= "Class Selection", default_open=True, tag="class_selection_header", show=False):
                dpg.add_text("New Category", tag="Add Class")
                dpg.bind_item_font(dpg.last_item(), bold_font)
                with dpg.group(horizontal=True, width=0, height=BTN_HEIGHT): 
                    dpg.add_input_text(label='', tag='Create New Class', height=BTN_HEIGHT)  
                    dpg.add_button(label="Create", height=BTN_HEIGHT, callback=create_class_callback)
                    dpg.bind_item_theme(dpg.last_item(), 'okay_button_theme')
                    dpg.bind_item_font(dpg.last_item(), button_font)

                dpg.add_text("Show Category")
                dpg.bind_item_font(dpg.last_item(), bold_font)
                with dpg.group(horizontal=True, height=BTN_HEIGHT ):                       
                    dpg.add_combo(label='', items=(), default_value='', callback=class_show_callback, tag="class_show_dropdown")
                dpg.add_spacer(height=3)
                dpg.add_button(label='Make it Instance',height=BTN_HEIGHT, callback=make_instance_callback, tag='make_instance', show=False)
                dpg.bind_item_theme(dpg.last_item(), 'okay_button_theme')
                dpg.bind_item_font(dpg.last_item(), button_font)
                dpg.add_spacer(height=5)
                with dpg.group(horizontal=True, height=BTN_HEIGHT ):    
                    dpg.add_button(label="Delete", height=BTN_HEIGHT, callback=delete_class_callback)
                    dpg.bind_item_theme(dpg.last_item(), 'delete_button_theme')
                    dpg.bind_item_font(dpg.last_item(), button_font)
                    dpg.add_checkbox(label="Show all", callback=show_all_class_callback, default_value=True, show=False)

                dpg.add_spacer(height=5)
                with dpg.group(horizontal=True, height=BTN_HEIGHT , tag="Stroke Slider Selection", show=False):
                    dpg.add_text('Stroke Size:')
                    dpg.add_slider_int(label="", default_value=stroke, max_value=8, min_value=1, callback=stroke_callback, tag="Stroke Slider", width=150)
            with dpg.collapsing_header(label= "Instance Selection", tag="instance_selection_header", default_open=True, show=False):
                with dpg.group(horizontal=True, height=30):
                    dpg.add_text("Category:  ", color=[255, 255, 255])
                    dpg.bind_item_font(dpg.last_item(), bold_font)
                    dpg.add_spacer(width=4)
                    dpg.add_text("", tag="instance_class_label" )
                dpg.add_spacer(height=4)
                with dpg.group(horizontal=True, height=30, show=True):                       
                    dpg.add_listbox(label="", items=(), default_value="",num_items=4, width=0, tag="instance_listbox")
                    
                dpg.add_spacer(height=4)
                with dpg.group(horizontal=True, height=30):
                    dpg.add_button(label="New Instance",  height=30, callback=create_instance_callback)
                    dpg.bind_item_theme(dpg.last_item(), 'okay_button_theme')
                    dpg.bind_item_font(dpg.last_item(), button_font)
                    dpg.add_text("", tag="instance_id_label")
                    dpg.bind_item_font(dpg.last_item(), button_font)
            dpg.add_spacer(height=5)
            dpg.add_checkbox(label="Show instance BBoxes", callback=show_ins_bbox_callback, tag="show_instance_bbox", show=False)
            dpg.add_spacer(height=10)
            dpg.add_image("icon_image", tracked=False)
    
    dpg.add_separator()
    dpg.add_spacer(height=3)
    with dpg.group(horizontal=True, width=150, tag="Second Panel"):
        dpg.add_combo((annotation_task_list), label="", default_value=" ", callback=task_select_callback, tag="task_dropdown")
        dpg.add_text("Select a task at the begining.", tag=annotation_ratio_btn_list[0])
        dpg.add_radio_button(("Select", "Click"), show=False, callback=annotation_mode_callback, horizontal=True, tag=annotation_ratio_btn_list[1])
        dpg.add_radio_button(("Select", "Rectangle"), show=False, callback=annotation_mode_callback, horizontal=True, tag=annotation_ratio_btn_list[2])
        dpg.add_radio_button(segmentation_nosuper_op_list, show=False, callback=annotation_mode_callback, horizontal=True, tag=annotation_ratio_btn_list[3])
        dpg.add_radio_button(segmentation_nosuper_op_list, show=False, callback=annotation_mode_callback, horizontal=True, tag=annotation_ratio_btn_list[4])
        dpg.add_radio_button(segmentation_nosuper_op_list, show=False, callback=annotation_mode_callback, horizontal=True, tag=annotation_ratio_btn_list[5])
        dpg.add_button(label="Create", height=BTN_HEIGHT, callback=polygon_finish_callback, tag="polygon_finish", show=False)
        dpg.bind_item_theme(dpg.last_item(), 'okay_button_theme')
        dpg.bind_item_font(dpg.last_item(), button_font)
        dpg.add_button(label="Delete", height=BTN_HEIGHT, callback=annotation_delete_callback, tag="annotation_delete", show=False)
        dpg.bind_item_theme(dpg.last_item(), 'delete_button_theme')
        dpg.bind_item_font(dpg.last_item(), button_font)
        dpg.add_button(label="Cancel", height=BTN_HEIGHT, callback=polygon_finish_callback, tag="polygon_cancel", show=False)
        dpg.bind_item_theme(dpg.last_item(), 'cancel_button_theme')
        dpg.bind_item_font(dpg.last_item(), button_font)
        with dpg.group(horizontal=True, width=30, tag="Class Assign", show=False):
            dpg.add_text(' Assign to ')
            dpg.add_combo(label='', width=5, items=(), default_value='', callback=class_assign_callback, tag="class_assign_dropdown")     

        dpg.add_button(label="AI Annotate", height=BTN_HEIGHT, callback=ai_annotate_callback, tag="ai_annotate_btn", show=False, pos=[1080, 795])
        dpg.bind_item_theme(dpg.last_item(), 'magic_button_theme')
        dpg.bind_item_font(dpg.last_item(), button_font)
        dpg.add_checkbox(label="All", callback=ai_annotate_all_callback, tag="ai_annotate_all", show=False, pos=[1240, 797])

    # check out simple module for details
    # with dpg.popup(dpg.last_item(), mousebutton=dpg.mvMouseButton_Left, modal=True, tag="modal_id"):
    #     dpg.add_button(label="Close", callback=lambda: dpg.configure_item("modal_id", show=False))

    for i in annotation_ratio_btn_list:
        dpg.bind_item_font(i, bold_font)

with dpg.window(label="Loading", tag="LoadingWindow", modal=True, no_close=False, 
                no_bring_to_front_on_focus=False, show=False, no_title_bar=True, 
                no_background=True,popup=True, pos = [WINDOW_WIDTH//2-70, WINDOW_HEIGHT//2-50],
                no_move=True, no_resize=True, no_collapse=True, width=200):
    with dpg.group(horizontal=True):
        dpg.add_spacer(width=20)
        dpg.add_loading_indicator(tag='Loading Indicator', circle_count=6)
    dpg.add_text("Loading...", tag='Loading Message')
    dpg.bind_item_font(dpg.last_item(), bold_font)
                
dpg.create_viewport(title='picA - Annotation Tool', width=WINDOW_WIDTH, height=WINDOW_HEIGHT)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.set_primary_window("Image", True)

panel_xy = dpg.get_item_pos('Main Panel') # at this time the panel is not rendered yet (and will return (0, 0))

# Main loop
while dpg.is_dearpygui_running():
    if refresh:
        if image is not None:
            tmp_image, resized_image = kernel.render_annotation(original_image, image, show_all=show_all, show_bbox=show_bbox, stroke=stroke)
            if resized_image is not None:
                image = resized_image

            lx = int(workspace[4])
            ly = int(workspace[5])
            rx = int(image.shape[1]) if tmp_image.shape[1] < workspace[2] else workspace[2]
            ry = int(image.shape[0]) if tmp_image.shape[0] < workspace[3] else workspace[3]
            tmp_image = tmp_image[ly:ly+ry, lx:lx+rx, :]
        else:
            tmp_image = np.zeros((1, 1, 4), dtype=np.uint8)
            
        image_dpg = dpg_image(tmp_image)

        if dpg.does_item_exist("source_image"):
            dpg.set_value("source_image", image_dpg)

        if kernel.select is not None:
            dpg.show_item('image_hover_info_dialog')
        else:
            dpg.hide_item('image_hover_info_dialog')

        refresh = False
        # print('Rendering...')

    # initialize workspace
    if panel_xy[1] == 0:
        panel_xy = dpg.get_item_pos('Main Panel')
        workspace = [panel_xy[0], panel_xy[1], 0, 0, 0, 0]

    dpg.render_dearpygui_frame()

quit()
dpg.destroy_context()