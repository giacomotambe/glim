![GLIM](docs/assets/logo2.png "GLIM Logo")

## Introduction

**GLIM** is a versatile and extensible range-based 3D mapping framework.

- ***Accuracy:*** GLIM is based on direct multi-scan registration error minimization on factor graphs that enables to accurately retain the consistency of mapping results. GPU acceleration is supported to maximize the mapping speed and quality.
- ***Easy-to-use:*** GLIM offers an interactive map correction interface that enables the user to manually correct mapping failures and easily refine mapping results.
- ***Versatility:*** As we eliminated sensor-specific processes, GLIM can be applied to any kind of range sensors including:
    - Spinning-type LiDAR (e.g., Velodyne HDL32e and Ouster OS1-32)
    - Non-repetitive scan LiDAR (e.g., Livox Avia and MID360)
    - Solid-state LiDAR (e.g., Intel Realsense L515)
    - RGB-D camera (e.g., Microsoft Azure Kinect)
- ***Extensibility:*** GLIM provides the global callback slot mechanism that allows to access the internal states of the mapping process and insert additional constraints to the factor graph. We also release [glim_ext](https://github.com/koide3/glim_ext) that offers example implementations of several extension functions (e.g., explicit loop detection, LiDAR-Visual-Inertial odometry estimation).

**Documentation: [https://koide3.github.io/glim/](https://koide3.github.io/glim/)**
**Docker hub:** [koide3/glim_ros2](https://hub.docker.com/repository/docker/koide3/glim_ros2/tags)
**Related packages:** [gtsam_points](https://github.com/koide3/gtsam_points), [glim](https://github.com/koide3/glim), ~~[glim_ros1](https://github.com/koide3/glim_ros1),~~ [glim_ros2](https://github.com/koide3/glim_ros2), [glim_ext](https://github.com/koide3/glim_ext)

Tested on Ubuntu 22.04 / 24.04 with CUDA 12.2 / 12.6 / 13.1, and NVIDIA Jetson Orin (Jetpack 6.1).

If you find this package useful for your project, please consider leaving a comment [here](https://github.com/koide3/glim/issues/19). It would help the author receive recognition in his organization and keep working on this project.

[![Build](https://github.com/koide3/glim/actions/workflows/build.yml/badge.svg)](https://github.com/koide3/glim/actions/workflows/build.yml)
[![ROS2](https://github.com/koide3/glim_ros2/actions/workflows/build.yml/badge.svg)](https://github.com/koide3/glim_ros2/actions/workflows/build.yml)
[![EXT](https://github.com/koide3/glim_ext/actions/workflows/build.yml/badge.svg)](https://github.com/koide3/glim_ext/actions/workflows/build.yml)

## Updates

- 2026/01/24 : v1.2.0 released. Added Support for both **GTSAM 4.2a9** and **GTSAM 4.3a0**, and **CUDA 13.1**. Added intensity visualization support.
- 2025/06/15 : The base GTSAM version has been changed. Make sure you have rebuilt and installed **GTSAM 4.3a0** and **gtsam_points 1.2.0**.

## Dependencies
### Mandatory
- [Eigen](https://eigen.tuxfamily.org/index.php)
- [nanoflann](https://github.com/jlblancoc/nanoflann)
- [GTSAM](https://github.com/borglab/gtsam)
- [gtsam_points](https://github.com/koide3/gtsam_points)

### Optional
- [CUDA](https://developer.nvidia.com/cuda-toolkit)
- [OpenCV](https://opencv.org/)
- [OpenMP](https://www.openmp.org/)
- [ROS/ROS2](https://www.ros.org/)
- [Iridescence](https://github.com/koide3/iridescence)

## Gallery

See more at [Video Gallery](https://github.com/koide3/glim/wiki/Video-Gallery).

| Mapping with various range sensors | Outdoor driving test with Livox MID360 |
|---|---|
|[<img width="480" src="https://github.com/user-attachments/assets/95e153cd-1538-4ca6-8dd0-691e920dccd9">](https://www.youtube.com/watch?v=_fwK4awbW18)|[<img width="480" src="https://github.com/user-attachments/assets/6b337369-a32c-4b07-b0e0-b63f6747cdab">](https://www.youtube.com/watch?v=CIfRqeV0irE)|

| Manual loop closing | Merging multiple mapping sessions |
|---|---|
|![Image](https://github.com/user-attachments/assets/0f02950a-6b7b-437c-a100-21d6575f7c93)|![Image](https://github.com/user-attachments/assets/c77cca29-921b-4e1c-9583-2b962ccda2cb)|

| Object segmentation and removal |  |
|---|---|
|![Image](https://github.com/user-attachments/assets/fd1038e7-c33d-44b1-86f9-8e6474c04210)| |

## Estimation modules

GLIM provides several estimation modules to cover use scenarios, from robust and accurate mapping with a GPU to lightweight real-time mapping with a low-specification PC like Raspberry Pi.

![modules](docs/assets/module.png)

## Thirdparty works using GLIM

If you are willing to add your work here, feel free to let me know in [this thread](https://github.com/koide3/glim/issues/19) :)

- [kamibukuro5656/MapCleaner_Unofficial](https://github.com/kamibukuro5656/MapCleaner_Unofficial)

---

## Dynamic Object Rejection (JO extension)

This fork adds a **dynamic object rejection** pipeline that removes moving objects (people, vehicles, forklifts …) from each scan before it is fed to the SLAM front-end, preventing ghost artifacts in the map.

### Architecture overview

```
PreprocessedFrame
       │
       ▼
 ┌─────────────┐      wall voxelmap
 │  WallFilter │ ─────────────────────────────────┐
 └─────────────┘                                  │
       │ WallFilterResult                         │
       ▼                                          ▼
 ┌──────────────────────┐          ┌──────────────────────────┐
 │ DynamicClusterExtrac │          │  WallBBoxRegistry        │
 │ -tor  (DBSCAN + KF + │          │  (persistent wall OBBs)  │
 │  EMA tracking)       │          └──────────────────────────┘
 └──────────────────────┘
       │ vector<BoundingBox>
       ▼
 ┌─────────────────────────┐
 │ DynamicObjectRejection  │
 │ CPU  (per-voxel scorer) │
 └─────────────────────────┘
       │
   ┌───┴────────────┐
   ▼                ▼
static_frame   dynamic_frame
  (→ SLAM)      (discarded / visualised)
```

### 1. WallFilter
Voxelises the current scan and fits planar surfaces via **RANSAC**.  
Voxels belonging to walls, floor, or ceiling are tagged `is_wall = true` and bypass the dynamic scorer unconditionally.

Config: `config_wall_filter.json`

| Parameter | Default | Description |
|---|---|---|
| `voxel_resolution` | 0.25 m | Voxel size |
| `ransac_max_iterations` | 100 | |
| `ransac_inlier_threshold` | 0.07 m | |
| `ransac_min_inliers` | 30 | |
| `wall_vertical_angle_deg` | 5° | Planes within this angle of vertical are walls |
| `floor_ceiling_angle_deg` | 0° | Planes within this angle of horizontal are floor/ceiling |
| `floor_min_inliers` | 250 | Minimum inliers to accept a floor/ceiling plane |
| `max_planes` | 8 | Maximum planes extracted per frame |
| `wall_bbox_max_aspect_ratio` | 5.0 | Aspect-ratio cap for wall OBBs |

### 2. WallBBoxRegistry
Maintains a **persistent registry** of oriented bounding boxes (OBBs) for wall segments across frames. Each frame's wall OBBs are matched to the registry via 3-D SAT IoU; matched boxes are merged (weighted average + SLERP rotation), unmatched ones are added. Stale entries are pruned after `max_missed_frames` frames.

Config: `config_wall_registry.json`

| Parameter | Default | Description |
|---|---|---|
| `overlap_threshold` | 0.3 | IoU above which boxes are merged |
| `merge_weight` | 0.3 | Weight of new observation during merge |
| `enable_expiry` | true | Remove boxes unseen for `max_missed_frames` |
| `max_missed_frames` | 30 | |
| `min_normal_dot` | 0.9 | Minimum normal alignment for merge candidates |
| `max_center_distance` | 5.0 m | Maximum center distance for merge candidates |

### 3. DynamicClusterExtractor
Runs **DBSCAN** on non-wall voxel centroids, then tracks each cluster across frames with a **constant-velocity Kalman filter** and classifies it via an **EMA dynamic score**.

**Pipeline per frame:**
1. DBSCAN on non-wall voxel centroids → raw clusters
2. Build OBBs per cluster, apply dimensional filters
3. Containment-based bbox merging
4. `update_tracks()` — nearest-neighbour matching with Mahalanobis gating; Kalman predict/update per track
5. Velocity-based + EMA-score classification with hysteresis
6. Distance-based static bias (far objects biased toward static)

Config: `config_dynamic_cluster_extractor.json`

**DBSCAN**

| Parameter | Default | Description |
|---|---|---|
| `eps_voxel_factor` | 1.2 | `eps = factor × voxel_resolution` |
| `min_pts` | 2 | Minimum neighbours for a core point |
| `knn_max_neighbors` | 32 | KNN search cap |
| `min_cluster_voxels` | 3 | Discard clusters with fewer voxels |
| `min_points_for_bbox` | 10 | Minimum raw points to generate an OBB |

**Bounding-box dimensional filter**

| Parameter | Default | Description |
|---|---|---|
| `bbox_min_extent` | 0.1 m | Minimum size along any axis |
| `bbox_max_extent` | 3.0 m | Maximum size along any axis |
| `bbox_min_volume` | 0.005 m³ | |
| `bbox_max_volume` | 6.0 m³ | |

**Tracking**

| Parameter | Default | Description |
|---|---|---|
| `track_match_distance` | 1.5 m | Max predicted-center distance for association |
| `track_match_distance_strict` | 0.5 m | Distance below which IoU gating is skipped |
| `track_match_iou` | 0.2 | Minimum IoU for association |
| `track_max_missed` | 10 | Frames before track deletion |

**Kalman filter (per track)**

| Parameter | Default | Description |
|---|---|---|
| `kf_q_pos` | 0.02 m²/frame² | Process noise — position |
| `kf_q_vel` | 0.20 (m/frame)²/frame | Process noise — velocity |
| `kf_r_pos` | 0.09 m² | Measurement noise (~0.3 m std-dev) |
| `kf_mahal_gate` | 9.21 | χ² gate df=3 (99 % confidence) |
| `kf_vel_damping` | 1.0 | Velocity multiplied by this each predict step |

**Classification**

| Parameter | Default | Description |
|---|---|---|
| `velocity_beta` | 0.8 | LP smoothing for velocity (0=instant, 1=frozen) |
| `velocity_static_threshold` | 0.03 m/frame | Speed below which a track is slow |
| `velocity_static_frames` | 3 | Consecutive slow frames → force static |
| `velocity_dynamic_threshold` | 0.2 m/frame | Speed above which a track is definitely dynamic |
| `motion_scale_factor` | 2.0 | Scales thresholds with ego-motion magnitude |
| `ema_alpha` | 0.5 | EMA smoothing for dynamic score |
| `dynamic_score_threshold` | 0.5 | Score ≥ threshold → dynamic |
| `min_track_age` | 3 | Young tracks stay dynamic (uncertain) |
| `hysteresis_dynamic_n` | 2 | Frames to flip static → dynamic |
| `hysteresis_static_m` | 5 | Frames to flip dynamic → static |
| `distance_static_bias_start` | 12 m | Bias start distance |
| `distance_static_bias_end` | 30 m | Bias end distance (full bias) |
| `distance_static_bias_max` | 0.25 | Maximum additive bias on static score |

### 4. DynamicObjectRejectionCPU
Per-voxel scorer that compares the current voxelmap against a ring-buffer of past voxelmaps. Each voxel receives a **composite dynamic score** from:

- **Shift score** — centroid displacement between frames
- **Mahalanobis score** — covariance-weighted displacement
- **Neighbor score** — fraction of neighbouring voxels that are dynamic
- **Cluster score** — whether the voxel falls inside a confirmed dynamic cluster OBB
- **History score** — fraction of past frames in which this voxel location was dynamic

Three tiers allow different thresholds depending on context (inside confirmed cluster bbox / dynamic memory only / unconstrained).

Config: `config_dynamic_object_rejection.json`

| Parameter | Default | Description |
|---|---|---|
| `dynamic_score_threshold` | 0.40 | Base classification threshold |
| `tier1_threshold_factor` | 0.55 | Multiplier for voxels inside confirmed dynamic bbox |
| `memory_threshold_factor` | 0.70 | Multiplier for voxels with dynamic history only |
| `unconstrained_threshold_factor` | 3.0 | Multiplier for isolated voxels (reduces false positives) |
| `w_shift` | 1.0 | Weight — shift score |
| `w_mahalanobis` | 0.0 | Weight — Mahalanobis score |
| `w_cluster` | 0.2 | Weight — cluster score |
| `w_neighbor` | 0.1 | Weight — neighbor propagation |
| `w_history` | 0.35 | Weight — history score |
| `frame_num_memory` | 5 | Past frames kept in ring buffer |
| `history_factor` | 0.25 | Scale factor applied to history score |
| `points_limit` | 0.3 | Min fraction of voxel points needed to trigger |
| `dynamic_cluster_ratio` | 0.4 | Fraction of dynamic voxels to propagate whole cluster |
| `cluster_propagation_threshold` | 0.3 | Score threshold for cluster propagation |
| `num_threads` | 4 | OpenMP threads for voxel scoring |

### Source files

| File | Description |
|---|---|
| `src/glim/dynamic_rejection/wall_bbox.cpp` | WallBBoxRegistry — 3-D SAT IoU, merge, expiry |
| `src/glim/dynamic_rejection/cluster_extractor.cpp` | DBSCAN, Kalman tracking, EMA classification |
| `src/glim/dynamic_rejection/dynamic_object_rejection_cpu.cpp` | Per-voxel scorer, tier thresholds, history ring-buffer |
| `src/glim/dynamic_rejection/dynamic_bounding_box_rejection.cpp` | Async wrapper / ROS2 integration |
| `src/glim/dynamic_rejection/voxel_filtering.cpp` | Voxel utility helpers |
| `src/glim/dynamic_rejection/transformation_kalman_filter.cpp` | Ego-motion Kalman filter (pose smoothing) |

---

## License

This package is released under the MIT license. For commercial support, please contact ```k.koide@aist.go.jp```.

If you find this package useful for your project, please consider leaving a comment [here](https://github.com/koide3/glim/issues/19). It would help the author receive recognition in his organization and keep working on this project. Please also cite the following paper if you use this package in your academic work.

## Related work

Koide et al., "GLIM: 3D Range-Inertial Localization and Mapping with GPU-Accelerated Scan Matching Factors", Robotics and Autonomous Systems, 2024, [[DOI]](https://doi.org/10.1016/j.robot.2024.104750) [[Arxiv]](https://arxiv.org/abs/2407.10344)

The GLIM framework involves ideas expanded from the following papers:
- (LiDAR-IMU odometry and mapping) "Globally Consistent and Tightly Coupled 3D LiDAR Inertial Mapping", ICRA2022 [[DOI]](https://doi.org/10.1109/ICRA46639.2022.9812385)
- (Global registration error minimization) "Globally Consistent 3D LiDAR Mapping with GPU-accelerated GICP Matching Cost Factors", IEEE RA-L, 2021, [[DOI]](https://doi.org/10.1109/LRA.2021.3113043)
- (GPU-accelerated scan matching) "Voxelized GICP for Fast and Accurate 3D Point Cloud Registration", ICRA2021, [[DOI]](https://doi.org/10.1109/ICRA48506.2021.9560835)

## Contact
[Kenji Koide](https://staff.aist.go.jp/k.koide/), k.koide@aist.go.jp<br>
National Institute of Advanced Industrial Science and Technology (AIST), Japan

