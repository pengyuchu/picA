<div align="center">
  <p>
    <a href="https://github.com/pengyuchu/picA" target="_blank">
      <img width="100%" src="https://github.com/pengyuchu/picA/assets/7012874/df958eb7-8766-4b5c-8d4a-5cb6133a7795"></a>
  </p>
<!-- <div> -->
    <!-- <a href=""><img src="" alt="Docker Pulls"></a> -->
    <!-- <br> -->
<!-- </div> -->
</div>

<div align="center">
[PicA](https://github.com/pengyuchu/picA) is an AI-powered and user-friendly image annotation tool designed to simplify the process of preparing image datasets for machine learning projects. With a focus on versatility and efficiency, picA offers a range of annotation features tailored to different use cases, including object counting, object detection, instance segmentation, semantic segmentation, and panoptic segmentation.
</div>
</br>

<div align="center">
    <img width="100%" src="https://github.com/pengyuchu/picA/assets/7012874/7d1cc933-5cc5-45f3-9652-0d73382ae0e3" />
</div>

## :star2: Key Features :star2:

- **Versatile Annotation**

    `picA` supports a variety of annotation tasks, allowing you to seamlessly annotate images for different purposes, from simple object counting to complex instance and semantic segmentation.

- **Deep Learning Integration**

    Harness the capabilities of deep learning models to enhance your annotation process. `picA` supports integration with pre-trained models, enabling you to accelerate and improve annotation accuracy.

- **Custom Model Deployment** [TBD]

    Utilize your own custom deep learning models within `picA`. Tailor your annotations to your project's unique requirements, ensuring precise and reliable results.

- **Export in COCO Format** 

    Easily export your annotations in the COCO (Common Objects in Context) format, a widely used standard in the computer vision community. Your annotated data is ready for integration into your machine learning pipeline.

## Roadmap :eyes:
### Wait to Test
- [x] COCO format annotations export.
- [ ] Category visibility control.
- [ ] Customized model integration.

### Short-Term Goals
- [ ] Nested category supports.
- [ ] Keypoint annotation supports.
- [ ] Rotated bounding box annotation supports.

## Installlation :scream:

See below for a quickstart installation or it's recommended to directly pulling a docker file.

### Code Install

Pip installation need Python>=3.8 and requirements below:
* Python >= 3.8
* pip >= 22.0
* PyTorch >= 1.10

```bash
cd /path/to/picA
pip install -r requirements.txt
```

### Docker Install

TBD.

## Getting Started :runner:

### Launch

`picA` may be launched directly in the Command Line Interface (CLI):

```bash
cd /path/to/picA
python main.py
```

### Create or Import a project
To create a new project, click `File -> New Project` to select a folder that contains image data. PicA officially supports `jpg`, `jpeg`, `png`, and `bmp` image formats.

```
Project Name
|--- image01.jpg
|--- image02.jpg
|--- ...
```
The project will be saved in the selected foler and the results look like:
```
Project Name
|--- image01.jpg
|--- image02.jpg
|--- ...
|--- annotations
|.. |--- annotations.json
|.. |--- masks
|.. |--- color_masks
|.. |.. |--- image01.png
|.. |.. |--- image02.png
|.. |.. |--- ...
```

When importing an existing project with `File -> Import Project`, ensure that you select the `Project Name` folder and verify that all images and annotations are located within the same `Project Name` directory.

If **creating** or **importing** a project encounters issues, a simple solution is to relaunch the application.

### Merge two projects
Annotation integration is also available. Upon opening the **Project 1** window, you can opt to `Import Project` to merge **Project 2** with the existing one. However, please exercise caution and consider the following guidelines:

- Merging is supported only for projects with the same annotation tasks.
- Ensure that all images are situated within the directory of the **Project 1**.
- In cases of conflicts, such as annotations on the same image in both projects, picA will prioritize retaining annotations from **Project 1**.

All annotations will be preserved within the context of the **Project 1**, providing a consolidated and organized annotation repository.

### Operation
In `picA`, interactions can be categorized into two distinct operations: **File or Category Selection** and **Annotation Manipulation**.

#### File or Category Selection
This operation is accessible through the side panel. Here, users can seamlessly navigate between different files and efficiently manage categories. Within this panel, you have the capability to create, delete, and modify categories. Additionally, the ability to create instances offers for **Instance and Panoptic Segmentation**.
 

#### Annotation Manipulation
Found at the bottom panel, this operation encompasses the core of annotation activities. It's divided into two primary modes: `Select` and `Draw`. The `Select` mode empowers users to highlight, modify, and delete existing annotations, as well as reassign them to different categories. On the other hand, the `Draw` mode employs distinct terminology tailored to specific annotation tasks, such as `Click`, `Rectangle`, and `Polygon`.


### :tada: Smart Annotation :tada:
Under `Smart Annotation` menu, `picA` supports `Superpixel` and `AI model` two ways to help the annotation tasks (`Superpixel` only supports segmentation tasks.). By checking either options,  action buttons will appear at the **botton panel**.

`picA` officially supports annotation tasks for **Obejct Counting**, **Object Detection**, **Instance Segmentation**, **Semantic Segmentation** and **Panoptic Segmentation**.

<div align="center">
    <img width="1024" src="https://github.com/pengyuchu/picA/assets/7012874/178348fa-051d-4ce4-a6bc-1eb61888aad3">
</div>

### Save
At present, `picA` does not feature an autosave function. Users are required to save their projects manually, and the progress will also be automatically saved upon exiting `picA`.

### Shortcut :page_with_curl:

| Key  | Action
|-------- | --------------------- | 
| :leftwards_arrow_with_hook: or `Space` | Polygon Create |
| `Backspace` | Detete |
| `Esc` | Cancel |
| `w`, `s`, `a`, `d` | Move Image |
| :arrow_up:, :arrow_down: | Zoom In / Out |
| :arrow_left:, :arrow_right: | File Selection |
 

## FAQ :grey_question:

For picA bug reports and feature requests please visit [GitHub Issues](https://github.com/pengyuchu/picA/issues).

## Citation
If you use this annotation tool in your research, please cite this project.

```
@software{PicA_Image_Annotation_Toolbox_2023,
  author = {Pengyu Chu},
  doi = {10.5281/zenodo.1234},
  month = {07},
  title = {{PicA: An AI-powered Image Annotation Toolbox}},
  url = {https://github.com/pengyuchu/picA},
  version = {0.1.1},
  year = {2023}
}
```

## License

This project is released under the [GPL-3.0 license](https://github.com/pengyuchu/picA/blob/main/LICENSE).

