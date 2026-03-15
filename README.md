# UEGaussianSplatting (WIP)

> Still Work In Progress! Not Done Yet!

## Overview
Loading and rendering highly detailed large scale assets in 3D software is always a big challenge.
Unreal Engine has the feature like Nanite for handling high poly/large scale mesh.
However, currently there isn't any solution for handling high density/large scale gaussian splats.
This project primarily aims to solve this issue though dynamic streaming methods.

## Temporary Result
- 1 million gaussian splats in 120 fps

![Temp](https://github.com/user-attachments/assets/71afa392-4e3d-49e1-ae3b-5cebe31621a8)

- Nanite-like gaussin splatting performance optimization solution

<img width="1594" height="641" alt="Nanite1" src="https://github.com/user-attachments/assets/0c028439-3bc7-4cb8-8e46-dc986a267675" />
<img width="1915" height="1025" alt="Nanite2" src="https://github.com/user-attachments/assets/71612017-103c-41b5-8240-fb13dd98f485" />



## To Do List
- [x] Import, process PLY file and save as uasset
- [x] Render splats at given postiion
- [x] Add efficient sorting algorith: Radix Sort
- [x] Render color and alpha blending
- [x] Hook render result into Unreal Engine's render pipeline: before post process
- [x] Speed up serialize and deserialize process: Bulk Data
- [x] Hierarchical Data Structure: Cluster + LOD
- [ ] Dynamic switching micro LOD
- [ ] Dynamic load and unload macro tiles
- [ ] Anti Aliasing
- [ ] Performance Tracking (insight, render pass)
- [ ] Support relight

## How To Use It
- Simply press the import button to import PLY file
<img width="903" height="303" alt="ImportPLY" src="https://github.com/user-attachments/assets/8192e65d-7f3e-4bc6-ade0-ca29159c900f" />

- Add Gaussian Splat Actor to the level
<img width="929" height="248" alt="GaussianSplatActor" src="https://github.com/user-attachments/assets/96ecbb2d-18f0-42a5-a6d2-32db1480743c" />

- Assign the imported PLY uasset to the Gaussian Splat Actor
<img width="927" height="475" alt="GaussianSplatActorAssign" src="https://github.com/user-attachments/assets/00083594-29b4-4706-90ce-af19e6ff6ba7" />

## Debug Console Command
- `gs.ShowClusterBounds 0`: disable Nanite cluster preview
- `gs.ShowClusterBounds 1`: enable Nanite cluster preview
- `gs.DebugForceLODLevel ?`: force render a specific LOD cluster for debugging purpose (can be 1,2,3,4...)
- `gs.MaxRenderBudget ?`: Limit the max number of visible splats(after culling) for saving VRAM. By default there is no limitaion. Set the max cap to decrease the VRAM usage (ex:3,000,000). The culling will start from the splats that is far from the camera


