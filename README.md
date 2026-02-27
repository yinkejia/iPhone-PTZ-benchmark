# CamDirector: Towards Long-Term Coherent Video Trajectory Editing
### [Paper](https://yinkejia.github.io/CamDirector-Project-Page/) | [Project Page](https://yinkejia.github.io/CamDirector-Project-Page/) | [Video](https://yinkejia.github.io/CamDirector-Project-Page/) | [Data](https://huggingface.co/datasets/yinkejia/iPhone-PTZ)
This repo contains the official evaluation code for iPhone-PTZ benchmark proposed in [CVPR 2026 | CamDirector: Towards Long-Term Coherent Video Trajectory Editing]. Benchmark data and our method's results can be downloaded from [here](https://huggingface.co/datasets/yinkejia/iPhone-PTZ).


> [**CamDirector: Towards Long-Term Coherent Video Trajectory Editing**](https://yinkejia.github.io/CamDirector-Project-Page/) <br> 
> [Zhihao Shi](https://scholar.google.com/citations?user=xFAV1X8AAAAJ&hl=en)<sup>1*</sup>, 
[Kejia Yin](https://yinkejia.github.io/)<sup>2*</sup>, 
[Weilin Wan](https://scholar.google.com/citations?user=y3MnZUEAAAAJ&hl=en)<sup>3</sup>, 
[Yuhongze Zhou](https://ca.linkedin.com/in/yuhongze-zhou-7b3838231)<sup>4</sup>, 
[Yuanhao Yu](https://scholar.google.com/citations?user=KM4V0a8AAAAJ&hl=en)<sup>1</sup>, 
[Xinxin Zuo](https://sites.google.com/site/xinxinzuohome/)<sup>5</sup>, 
[Qiang Sun](https://scholar.google.com/citations?user=f0V2fAYAAAAJ&hl=en)<sup>2,6</sup>, 
[Juwei Lu](https://scholar.google.ca/citations?user=Asz24wcAAAAJ&hl=en)<sup>2</sup><br>
> McMaster, UofT, HKU, McGill, Concordia, MBZUAI<br>
> <sup>*</sup> Equal contribution <br>
> CVPR 2026

## Set Up
```bash
# creat your python env and:
pip install -r requirements.txt
```

## Prepare data
Please download our processed benchmark data and our method's results from [here](https://huggingface.co/datasets/yinkejia/iPhone-PTZ). Then organize the folder structure as following:
### Dataset Structure

```text
./evaluation_benchmarks
├── iPhone/
│   ├── apple
|       ├── camera_poses
│           ├── source_c2ws.pt
│           ├── source_Ks.pt
│           ├── target_c2ws.pt
│           └── target_Ks.pt
|       ├── depth_maps
│           ├── frame_00000_depth.png
│           ├── frame_00001_depth.png
│           ├── frame_00002_depth.png
│           └── ...
|       ├── point_clouds
│           ├── sparse_pcds_src.saftetensors
|       ├── source_imgs
│           ├── frame_00000.png
│           ├── frame_00001.png
│           ├── frame_00002.png
│           └── ...
|       ├── target_imgs
│           ├── frame_00000.png
│           ├── frame_00001.png
│           ├── frame_00002.png
│           └── ...
|   ├── block
|   ├── ...
├── iPhone-PTZ/
│   ├── container
|       ├── camera_poses
│           ├── source_c2ws.pt
│           ├── source_Ks.pt
│           ├── target_c2ws.pt
│           └── target_Ks.pt
|       ├── depth_maps
│           ├── frame_00000_depth.png
│           ├── frame_00001_depth.png
│           ├── frame_00002_depth.png
│           └── ...
|       ├── point_clouds
│           ├── sparse_pcds_src.saftetensors
|       ├── source_imgs
│           ├── frame_00000.png
│           ├── frame_00001.png
│           ├── frame_00002.png
│           └── ...
|       ├── target_imgs
│           ├── frame_00000.png
│           ├── frame_00001.png
│           ├── frame_00002.png
│           └── ...
|   ├── corner
|   ├── ...
```
We provide our processed data for both [iPhone](https://arxiv.org/abs/2210.13445) and iPhone-PTZ benchmarks. For each scene, we provide the following:
* Camera poses (c2ws and Ks) for source and target video frames in opencv camera convention.
* Depth maps for source and target video frames, please refer to `vis_benchmark.py` on how to read them
* Sparse points clouds for source video frames. If you prefer use your own depth, you may use these sparse point clouds to align your depths to our camera system.

### Results Structure
```text
./output_frames
├── camdiretor/
│   ├── apple
│       ├── frame_00000.png
│       ├── frame_00002.png
│       ├── frame_00004.png
│       └── ...
|   ├── block
|   ├── ...
|   ├── container
|   ├── corner
|   ├── ...
```
We provide our method's results for you to reporduce the numbers in our paper, please refer to the [Evaluation](#evaluation) section on how to compute the metrics.

## Evaluation
After organize the data and results, you can simply run the following cmd to compute metrics, the results will be saved to `./evaluated_metrics` by default:
```bash
python compute_metrics.py
```

## Benchmark Visualization
We also provide a script for you to visualize our processed benchmarks, and you can refer to `vis_benchmark.py` on how to read our benchmark data as well. Simply run the following cmd (with benchmark and scene specified):
```bash
python vis_benchmark.py --benchmark iPhone-PTZ --scene container
```
This will save point cloud visualizations with cameras every 20 frames, in each `.ply` file you have:
* Point clouds for this source video frame
* All Source cameras (every 10 frames): each camera is represented by two vertices and an edge between them, the green vertice represents the camera position, and blue vertice with the edge represent the camera's looking direction.
* All Target cameras (every 10 frames): each camera is represented by two vertices and an edge between them, the red vertice represents the camera position, and yellow vertice with the edge represent the camera's looking direction.

## Training and Inference code
Due to IP-policy, we don't have plan to release training and inference code at the moment.

## Citation

If you find this repository useful for your research, please use the following:

```txt
```

## License

Both our code and dataset are released under the Apache 2.0 license.

## Acknowledgement

For iPhone benchmark, we reuse the calibrated colmap cameras from [Shape-of-Motion](https://github.com/vye16/shape-of-motion). 