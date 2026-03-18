# Nano Gaussian Splatting

![Image](https://github.com/user-attachments/assets/83ae3ba1-9414-4cb5-8703-19a15abddd21)

> [!NOTE]
> Source code will be released soon!

Supported Unreal Engine version: `UE5.6+`


## Overview
Rendering large scale gaussian splatting files is always a big challenge.
In order to render large scale gaussian splatting files in real-time, we will need to carefully handle VRAM, sorting millions of splats efficiently and drawing only the needed splats for the screen.

The plugin leverages techniques such as Nanite-style LOD Clusters, Screen-Space Error LOD Selection, Splat Compaction, Global Accumulator and GPU Radix Sort.
As a result users can render large scale gaussian splitting scenes efficiently.

## How To Use It
Video Tutorial

[![Watch the video](https://img.youtube.com/vi/5nyo6cfvio0/0.jpg)](https://www.youtube.com/5nyo6cfvio0)

- Download the plugin from Releases page
- Place the plugin inside Unreal Engine project's "Plugins" folder
- Press import button to import PLY file. A Gaussian Splat Asset will be created

![Image](https://github.com/user-attachments/assets/2d784b1e-c2a1-4cb3-a891-2e80b84e6c28)

- Drag the Gaussian Splat Asset directly into the level

![Image](https://github.com/user-attachments/assets/52d63cca-e850-478b-a6bb-ed88efce1f97)

- Enable/Disable Nanite through asset action if needed

![Image](https://github.com/user-attachments/assets/5b5d7d44-a142-46ae-bbb6-bcae1856cca3)


## Settings
<img width="828" height="453" alt="Image" src="https://github.com/user-attachments/assets/d7314cbf-b0b9-4748-ad5f-ba87375fbfef" />

| Category | Variable | Description |
| :--- | :--- | :--- |
| Quality | SH Order | sphere harmonics quality |
| Performance | Sort Every Nth Frame | adjust gaussian splats sorting speed |
| Performance | Enable Frustum Culling | enable/disable frustum culling |
| Performance | LOD Error Threshold | LOD cluster switching sensitivity |
| Rendering | Opacity Scale | adjust the opacity of gaussian splats|
| Rendering | Splat Scale | adjust the scale of gaussian splats|


## Debug Console Command
- `gs.ShowClusterBounds 1`: enable Nanite cluster preview (set to 0 to disable preview)
- `gs.DebugForceLODLevel ?`: force render a specific LOD cluster for debugging purpose (can be 1,2,3,4...)
- `gs.MaxRenderBudget ?`: limit the max number of visible splats(after culling) for saving VRAM. By default there is no limitaion(0). Set the max cap to decrease the VRAM usage (ex:3,000,000). The culling will start from the splats which are far from the camera

## Best Practice
![Image](https://github.com/user-attachments/assets/76089944-18a8-4a3d-b6f9-52045a72acb8)

A single big chunk of gaussian splatting file is not conducive to performance optimization. 
The splats outside the camera view can not be culled. 
Also, all the splats will be engaged in the sorting process all the time.

The ideal format is slicing a big chunk of gaussian splatting file into smaller pieces.
For example individual props for cinematic scenes or tiles for geo-spatial data.
