# Public Datasets — Paired Cascaded mmWave Radar + LiDAR for NVS Training

**Goal:** Find datasets with paired radar + LiDAR where the radar samples >5 Hz and/or the platform moves slowly enough that adjacent frames overlap (so a held-out test frame is bracketed by neighbouring train frames). ColoRadar (our current dataset) tops out at 5 Hz cascaded, which limits NVS interpolation quality (held-out test correlation ceiling ~0.59).

**Date compiled:** 2026-04-28

**Selection criteria:**
- Radar frame rate >= 5 Hz (preferred >= 10 Hz)
- Paired LiDAR (any rate)
- Cascaded TI MMWCAS preferred; 4D radar / single-chip / scanning radar acceptable as fallback
- Public download (research-only OK)
- Ego-motion small enough that consecutive frames overlap meaningfully

---

## Dataset Comparison Table

| Dataset | Year | Radar | Radar Hz | LiDAR | LiDAR Hz | Ego-motion | License | Size | Public? |
|---|---|---|---|---|---|---|---|---|---|
| ColoRadar | 2021 | TI MMWCAS (cascade, 12TX×16RX) | **5** | Ouster OS1-64 | 10 | Handheld + longboard | Apache-2.0 | ~52 sequences, ~2 hrs | Yes |
| K-Radar | 2022 | RETINA-4ST 4D radar (Smart Radar Systems) | **10** | Ouster OS2-64 + OS1-128 | 10 | Vehicle | CC-BY-NC-ND (data), Apache-2.0 (code) | 35K frames, 58 sequences | Yes |
| View of Delft (VoD) | 2022 | ZF FRGen21 3+1D | **~13** | Velodyne HDL-64 | 10 | Vehicle (urban) | Research-only (registration) | 8,600–10,000 frames | Yes (request) |
| Boreas | 2022 | Navtech CIR304-H scanning | **4** | Velodyne Alpha-Prime (128) | 10 | Vehicle | CC-BY 4.0 | 350+ km, multi-season | Yes (AWS Open Data) |
| RadarScenes | 2021 | 4× Continental 77 GHz (single-chip series) | **~17** (60 ms cycle) | None (no LiDAR) | – | Vehicle | CC-BY-NC-SA 4.0 | 4 hrs, 158 sequences | Yes |
| MulRan | 2020 | Navtech CIR204-H scanning | 4 | Ouster OS1-64 | 10 | Vehicle | Research (KAIST) | 350+ km | Yes |
| CRUW | 2021 | TI AWR1843 single-chip + camera (RA heatmaps) | **30** | None (camera-only fusion) | – | Vehicle | CC-BY-NC | 3.5 hrs, ~400K frames | Yes |
| aiMotive | 2022 | 2× long-range automotive radar (model not disclosed) | – (not specified) | 64-beam spinning | 10 | Vehicle | CC-BY-NC-SA 4.0 | 26,583 annotated frames, 176 scenes | Yes |
| Astyx HiRes2019 | 2019 | Astyx 6455 HiRes (single-chip, 5D points) | **10** | Velodyne VLP-16 | 10 | Vehicle | Research (request) | 546 frames | Yes (small) |
| Oxford Radar RobotCar | 2019 | Navtech CTS350-X scanning | **4** | 2× Velodyne HDL-32E | 20 | Vehicle | CC-BY-NC-SA 4.0 | 240K radar scans, 280 km | Yes |
| Ithaca-365 | 2022 | **None** (camera + LiDAR + GPS only) | – | Velodyne | 10 | Vehicle | Research-only | 15 km route × repeated | Yes |
| Zendar SPECTRA | 2020 | Zendar SAR (proprietary) | ~24 (related work) | Yes (paired) | – | Vehicle | Proprietary, mostly offline | – | **No longer available** |
| TJ4DRadSet | 2022 | Oculii Eagle 4D radar | – (not in docs) | 80-line LiDAR | 10 | Vehicle | NDA, university only, no commercial | 7,757 frames, 44 sequences | Request only; LiDAR release pending |
| nuScenes | 2019 | 5× Continental ARS 408-21 (single-chip 2D) | **13** | Velodyne HDL-32E | 20 | Vehicle | CC-BY-NC-SA 4.0 | 1,000 scenes × 20s | Yes |
| RADIal (Valeo) | 2022 | Valeo HD radar (12TX×16RX, 192 virtual) — raw ADC | – (not specified, inferred ~10) | 16-layer LiDAR | – | Vehicle | CC-BY-NC-SA 4.0 | 25K frames, 91 sequences, 2 hrs | Yes |
| MSC-RAD4R | 2023 | Oculii 4D radar (79 GHz, PDM FMCW) | **15** | LiDAR (multi-frame) | – | Vehicle | Research | 90K radar frames, 51.6 km | Yes |
| Snail-Radar | 2024 | Continental ARS548 + Oculii Eagle (4D) | – (reference 10 Hz) | Hesai Pandar XT32 | 10 | **Handheld + e-bike + SUV** | Research | 44 sequences, 8 routes | Yes |
| Radatron | 2022 | TI MMWCAS cascade (12TX×16RX, 192 virtual) | **10** | None (camera-only paired) | – | Vehicle | Research | 152K frames, 4.2 hrs | Yes |
| Dual Radar | 2025 | Arbe Phoenix + ARS548 (both 4D) | – (~10) | 80-line mechanical LiDAR | 10 | Vehicle | Research | 10K annotated frames, 151 sequences | Yes |
| RadarML / mmwcas | 2024+ | TI MMWCAS cascade | (early release; not yet specified) | Yes | – | **Indoor + outdoor + bike-mounted** | CC-BY | 29 hrs (i/q-1m subset) | Partial (contact) |
| MMVR | 2024 | 2× TI AWR2243 cascade (60–64 GHz) | – | None (RGB-D for GT) | – | Mounted-stationary, indoor | Research | 345K radar frames, 6 rooms | Yes |
| PixSet (LeddarTech) | 2021 | Single-chip radar (model not specified) | High (untriggered) | Pixell flash LiDAR | 10 | Vehicle | Research, free for academic | 29K frames, 97 sequences | Yes |
| RADIATE | 2020 | Navtech CTS350-X scanning | **4** | Velodyne HDL-32E | – | Vehicle | Research | 3 hrs, 200K labels | Yes |
| WaveRadar (synthetic) | – | Not located in this search; appears to be a private/synthetic name | – | – | – | – | – | – |

(Cells marked "–" mean the value was not stated in the documentation we could locate; treat with caution.)

---

## Per-Dataset Detail (with sources)

### 1. ColoRadar (UColorado, 2021) — **OUR CURRENT BASELINE**
- Paper: arXiv 2103.04510 — https://arpg.github.io/coloradar/
- Radar: TI MMWCAS-RF-EVM cascade (12 TX × 16 RX → 86 virtual elements)
- **Radar rate: 5 Hz** (Table 1 of paper)
- LiDAR: Ouster OS1-64 @ 10 Hz
- Ego-motion: handheld, longboard — slow + 6DoF
- License: Apache 2.0
- Size: 52 sequences, ~142 minutes, indoor/outdoor/mine/multi-use paths
- **Pros for NVS:** True cascade RF data; slow handheld motion; permissive license
- **Cons for NVS:** Only 5 Hz — frame-to-frame ego-motion 5–50 cm limits view interpolation

### 2. K-Radar (KAIST, 2022)
- Paper: arXiv 2206.08171 — https://github.com/kaist-avelab/K-Radar
- Radar: RETINA-4ST 4D imaging radar (Smart Radar Systems; 4DRT tensor 64×256×107°×37°)
- **Radar rate: 10 Hz** (per ar5iv full paper)
- LiDAR: Ouster OS2-64 (long-range) + OS1-128 (high-res) — both @ 10 Hz
- Ego-motion: vehicle (urban + highway + adverse weather)
- License: CC-BY-NC-ND (data), Apache-2.0 (code)
- Size: 35K frames, 58 sequences, 93.3K 3D bounding boxes
- **Pros for NVS:** Full 4D radar tensor (RAED) — analogous to our ADC; 10 Hz; dual high-res LiDAR ground truth
- **Cons for NVS:** Vehicle-only motion is fast (30+ km/h → 80+ cm between frames at 10 Hz); CC-BY-NC-ND restricts derivative datasets; not a TI MMWCAS chip — different antenna layout

### 3. View of Delft (TU Delft, 2022)
- Paper: IEEE RA-L 2022 — https://tudelft-iv.github.io/view-of-delft-dataset/
- Radar: ZF FRGen21 3+1D radar (range, az, el, Doppler — 192 virtual antennas reported in some sources)
- **Radar rate: ~13 Hz**
- LiDAR: Velodyne HDL-64 @ 10 Hz (LiDAR is the lead clock)
- Ego-motion: vehicle (urban Delft)
- License: research-only (registration required)
- Size: 8,600–10,000 synced frames
- **Pros:** Higher radar rate than ColoRadar; well-synced multi-modal
- **Cons:** Vehicle motion at urban speeds; not a TI cascade — proprietary ZF chip; only point cloud output, not raw RF

### 4. Boreas (UToronto, 2022)
- Paper: IJRR 2023, arXiv 2203.10168 — https://www.boreas.utias.utoronto.ca/
- Radar: Navtech CIR304-H scanning (mechanically rotating FMCW, NOT MIMO cascade)
- **Radar rate: 4 Hz** (per pyboreas DATA_REFERENCE.md: 400 azimuths, 4 Hz spin)
- LiDAR: Velodyne Alpha-Prime 128 @ 10 Hz
- Ego-motion: vehicle (multi-season Toronto loop)
- License: CC-BY 4.0
- Size: 350+ km, multi-season, AWS Open Data
- **Pros:** Permissive license; high-quality LiDAR
- **Cons:** Scanning radar (not MIMO/cascade) — fundamentally different data; 4 Hz is *worse* than ColoRadar; vehicle motion

### 5. RadarScenes (Aptiv/Continental, 2021)
- Paper: arXiv 2104.02493 — https://radar-scenes.com/
- Radar: 4× Continental 77 GHz series (single-chip), point-cloud output only
- **Radar rate: ~17 Hz** (60 ms cycle time)
- LiDAR: **None**
- Ego-motion: vehicle
- License: CC-BY-NC-SA 4.0
- Size: 4 hrs, 100 km, 158 sequences
- **Pros:** High radar rate
- **Cons:** **No LiDAR** → cannot construct mesh GT; single-chip not cascade; point clouds only

### 6. MulRan (KAIST, 2020)
- Paper: ICRA 2020 — https://sites.google.com/view/mulran-pr/dataset
- Radar: Navtech CIR204-H scanning
- **Radar rate: ~4 Hz** (typical Navtech)
- LiDAR: Ouster OS1-64 @ 10 Hz
- Ego-motion: vehicle
- License: research
- Size: 350+ km
- **Pros:** Same OS1-64 LiDAR as ColoRadar
- **Cons:** Scanning radar (not cascade); 4 Hz; vehicle

### 7. CRUW (UW, 2021)
- Paper: CVPRW 2021 / arXiv 2105.05207 — https://www.cruwdataset.org/
- Radar: TI AWR1843 single-chip → range-azimuth heatmaps (RAMaps)
- **Radar rate: 30 FPS**
- LiDAR: **None** (camera-fused)
- Ego-motion: vehicle (parking lot, campus, city, highway)
- License: CC-BY-NC
- Size: 3.5 hrs, ~400K frames
- **Pros:** Highest frame rate of any radar dataset surveyed (30 Hz)
- **Cons:** **No LiDAR**; single-chip 4-RX (3D, not 4D); RA heatmap only — no raw ADC, no elevation

### 8. aiMotive (2022)
- Paper: arXiv 2211.09445 — https://github.com/aimotive/aimotive_dataset
- Radar: 2× long-range automotive radar, 360° (model not publicly disclosed)
- Radar rate: not stated (radar is one of the faster sensors but exact Hz not in paper summary)
- LiDAR: 64-beam spinning @ 10 Hz
- Ego-motion: vehicle
- License: CC-BY-NC-SA 4.0
- Size: 176 scenes, 26,583 annotated frames
- **Cons:** Sparse documentation on radar; not cascade

### 9. Astyx HiRes2019 (Astyx, 2019)
- Paper: SDF 2019 — https://paperswithcode.com/dataset/astyx-hires2019
- Radar: Astyx 6455 HiRes (single-chip 5D output: x, y, z, Vr, mag)
- **Radar rate: 10 Hz**
- LiDAR: Velodyne VLP-16 @ 10 Hz
- Ego-motion: vehicle
- License: research (registration)
- Size: 546 frames (very small)
- **Cons:** Tiny — too small for NVS training; not cascade

### 10. Oxford Radar RobotCar (Oxford, 2019)
- Paper: ICRA 2020 / arXiv 1909.01300 — https://oxford-robotics-institute.github.io/radar-robotcar-dataset/
- Radar: Navtech CTS350-X scanning FMCW
- **Radar rate: 4 Hz** (typical Navtech CTS350)
- LiDAR: 2× Velodyne HDL-32E @ 20 Hz
- Ego-motion: vehicle (Oxford loop, 32 traversals)
- License: CC-BY-NC-SA 4.0
- Size: 240K radar scans, 2.4M LiDAR scans, 280 km
- **Cons:** Scanning radar; 4 Hz; vehicle speeds

### 11. Ithaca-365 (Cornell, 2022)
- Paper: CVPR 2022 — https://ithaca365.mae.cornell.edu/
- Radar: **None** (no radar at all)
- LiDAR: Velodyne @ 10 Hz; cameras @ 30 Hz
- **Cons:** Excluded — no radar

### 12. Zendar SPECTRA (Zendar, 2020)
- Paper: CVPRW 2020 (Mostajabi et al.)
- Radar: Zendar SAR (street-level synthetic-aperture radar)
- Radar rate: ~24 Hz reported in related work
- LiDAR: paired
- License: proprietary
- **Status: dataset no longer publicly available** (Zendar pivoted commercially)

### 13. TJ4DRadSet (Tongji, 2022)
- Paper: ITSC 2022 / arXiv 2204.13483 — https://github.com/TJRadarLab/TJ4DRadSet
- Radar: Oculii Eagle 4D radar (cascade)
- Radar rate: not in public docs
- LiDAR: 80-line mechanical (release pending)
- Ego-motion: vehicle
- License: NDA, college/university only, non-commercial; weekly request approval
- Size: 7,757 frames, 44 sequences
- **Cons:** Currently only 4D radar released; LiDAR is "ongoing"; restrictive NDA

### 14. nuScenes (Aptiv/Motional, 2019)
- Paper: CVPR 2020 / arXiv 1903.11027
- Radar: 5× Continental ARS 408-21 single-chip 2D
- **Radar rate: 13 Hz**
- LiDAR: Velodyne HDL-32E @ 20 Hz
- Ego-motion: vehicle
- License: CC-BY-NC-SA 4.0
- Size: 1,000 scenes × 20 s
- **Cons:** Single-chip 2D radar, very sparse point cloud (~50–80 points/scan); not cascade

### 15. RADIal (Valeo, 2022)
- Paper: CVPR 2022 / arXiv 2112.10646 — https://github.com/valeoai/RADIal
- Radar: Valeo HD radar (12 TX × 16 RX → 192 virtual antennas) — **raw ADC available**
- Radar rate: not explicitly stated (likely ~10 Hz)
- LiDAR: 16-layer
- Ego-motion: vehicle (highway, country, city)
- License: CC-BY-NC-SA 4.0
- Size: 91 sequences, 2 hrs, ~25K frames, 8,252 labelled
- **Pros:** Raw ADC + RAD tensor + RA + RD + point cloud (closest in modality to ColoRadar/our pipeline)
- **Cons:** Vehicle motion; LiDAR is only 16-layer; no exact Hz published

### 16. MSC-RAD4R (KAIST, 2023)
- Paper: IEEE RA-L 2023 — https://mscrad4r.github.io/
- Radar: Oculii 4D radar @ **15 Hz**
- LiDAR: present (90,864 4D radar frames vs 60,562 LiDAR frames)
- Ego-motion: vehicle (51.6 km, 100 min, day/night/snow/smoke)
- License: research
- **Pros:** Highest 4D-radar rate among automotive datasets surveyed
- **Cons:** Vehicle; not TI cascade

### 17. Snail-Radar (Wuhan U, 2024)
- Paper: IJRR 2025, arXiv 2407.11705 — https://snail-radar.github.io/
- Radar: Continental ARS548 + Oculii Eagle (both 4D point-cloud)
- Radar rate: not stated; reference trajectory at 10 Hz
- LiDAR: Hesai Pandar XT32 (32-beam) — paired with reference trajectory
- **Ego-motion: handheld + e-bike + SUV** (only dataset besides ColoRadar with handheld!)
- License: research
- Size: 44 sequences, 8 routes, multi-condition
- **Pros:** Handheld and e-bike platforms = slow motion (closest analogue to ColoRadar's collection style)
- **Cons:** No raw RF — only point cloud output; not TI cascade

### 18. Radatron (UIUC, 2022)
- Paper: ECCV 2022 — https://jguan.page/Radatron/
- Radar: TI-MMWCAS cascade (12 TX × 16 RX, 192 virtual antennas) — **same chip as ColoRadar**
- **Radar rate: 10 Hz** (paper states 10 fps)
- LiDAR: **None** — paired with ZED stereo camera only
- Ego-motion: vehicle
- License: research
- Size: 152K frames, 4.2 hrs
- **Pros:** Same TI MMWCAS cascade as ColoRadar at **2× the frame rate**; 1.2° azimuth, 5 cm range, 18° elevation
- **Cons:** **No LiDAR** — would need to add or skip mesh-supervised training; vehicle speeds

### 19. Dual Radar (2025)
- Paper: Nature Scientific Data 2025 — https://github.com/adept-thu/Dual-Radar
- Radar: Arbe Phoenix + ARS548 (both 4D)
- Radar rate: not explicitly stated (~10 Hz typical)
- LiDAR: 80-line mechanical
- Ego-motion: vehicle
- License: research
- Size: 10,007 frames, 151 sequences

### 20. RadarML / mmwcas (CMU, 2024+)
- Site: https://radarml.github.io/
- Radar: TI MMWCAS cascade — **raw I/Q time signals**
- **Ego-motion: indoor + outdoor + bike-mounted** (very promising for slow motion)
- LiDAR: included (rate not specified)
- License: CC-BY (i/q-1m subset)
- Size: 29 hrs of paired radar+LiDAR+camera; mmwcas-specific subset is "early development"
- Contact: Tianshu Huang (tianshu2@andrew.cmu.edu) for pre-release access
- **Pros:** Same TI MMWCAS chip as ColoRadar; bike-mounted/indoor → slow motion; raw I/Q exposed
- **Cons:** mmwcas subset still pre-release; frame rate not yet published

### 21. MMVR (MERL/Mitsubishi, 2024)
- Paper: ECCV 2024 / arXiv 2406.10708
- Radar: 2× TI AWR2243 mmWave cascade @ 60–64 GHz
- LiDAR: **None** (RGB-D ground truth)
- Ego-motion: **mounted-stationary** (rooms, indoor)
- License: research
- Size: 345K radar frames, 25 subjects, 6 rooms
- **Cons:** No LiDAR; static rig (no view variation across frames); indoor human-pose focus

### 22. PixSet (LeddarTech, 2021)
- Paper: ITSC 2021 / arXiv 2102.12010 — https://leddartech.com/solutions/leddar-pixset-dataset/
- Radar: present (model not specified; untriggered, "much higher" than 10 Hz)
- LiDAR: Pixell solid-state flash @ 10 Hz
- Ego-motion: vehicle (urban)
- License: free for academic use
- Size: 29K frames, 97 sequences
- **Cons:** Radar specs not detailed; not cascade

### 23. RADIATE (Heriot-Watt, 2020)
- Paper: arXiv 2010.09076
- Radar: Navtech CTS350-X scanning
- **Radar rate: 4 Hz**
- LiDAR: Velodyne HDL-32E
- Ego-motion: vehicle (adverse weather: rain/fog/snow/night)
- License: research

### 24. WaveRadar (synthetic)
- **Not located.** No verified public dataset under this name was found in our search. Likely refers to a synthetic / proprietary internal dataset, or our search keyword is slightly off. If you have a paper reference for WaveRadar, please supply it.

---

## TOP-3 RANKED RECOMMENDATIONS

The selection criteria are: (a) radar rate >= our current 5 Hz, (b) paired LiDAR (so we have mesh GT), (c) slow ego-motion (so consecutive frames overlap → bracketed test frames are achievable), (d) preferably TI MMWCAS cascade (matches our pipeline directly).

### #1 — RadarML / mmwcas (CMU, Tianshu Huang)
**Why:** This is the closest match to our exact use case — same TI MMWCAS cascade chip as ColoRadar, paired LiDAR, and crucially **bike-mounted / indoor / outdoor handheld** platforms (slow motion → small frame-to-frame ego-motion). Raw I/Q radar signals are exposed (matching our integrator/ADC pipeline). The CC-BY license on the i/q-1m subset is permissive.
**Caveats:** The MMWCAS-specific subset is pre-release as of mid-2024 — needs an email to the maintainer. Frame rate for the mmwcas subset is not yet on the project page.
**Action:** Email tianshu2@andrew.cmu.edu to request access and confirm radar Hz before committing.

### #2 — K-Radar (KAIST)
**Why:** **10 Hz** 4D radar tensor (RAED) — twice our current rate, and the 4DRT format is the closest existing analogue to our ADC-derived heatmap pipeline. Two paired Ouster LiDARs (OS2-64 + OS1-128) at 10 Hz give excellent mesh ground truth. Largest cascade-class dataset publicly released (35K frames, 58 sequences). Adverse weather coverage is a bonus for stress tests.
**Caveats:** Vehicle motion is fast (urban + highway) — even at 10 Hz, an 11 m/s vehicle moves 1.1 m between frames, which is *more* total ego-motion than our 5 Hz handheld at 0.5 m/s. To win on bracketing, you'd want to subset to slow urban segments. Also: CC-BY-NC-ND blocks remixing the data into a derived release. Radar is RETINA-4ST (Smart Radar Systems), not TI MMWCAS — antenna geometry differs, so material transfer between K-Radar and our cascade would not be apples-to-apples.

### #3 — Snail-Radar (Wuhan University)
**Why:** Only public dataset besides ColoRadar with **handheld** and **e-bike** platforms (small ego-motion!). Three platforms × 8 routes × multi-condition = 44 sequences. Paired Hesai Pandar XT32 LiDAR with 10 Hz reference trajectory. Multiple 4D radar models for cross-sensor generalization (ARS548 + Oculii Eagle).
**Caveats:** 4D radar output is point cloud only (no raw RF / no ADC), which is a step *down* in fidelity from ColoRadar. Radar Hz is not explicitly stated in the paper summary — needs checking. Not TI MMWCAS — generalization claims would be cross-vendor.

### Honourable Mentions

- **Radatron** would be #1 if it had LiDAR. It uses the *exact same TI MMWCAS chip* at *2× the frame rate* (10 Hz, 152K frames) — but it's paired only with stereo camera, no LiDAR mesh. If you can fuse Radatron's radar with externally-generated meshes (e.g., COLMAP from the ZED stereo), it becomes the strongest cascade-radar candidate by a wide margin.
- **MSC-RAD4R** has the highest verified 4D-radar rate (15 Hz) and paired LiDAR, but the radar is Oculii (not TI cascade) and ego-motion is vehicular.
- **RADIal** uniquely exposes raw ADC from a 192-virtual-antenna HD radar paired with LiDAR — modality-wise the closest to ColoRadar besides the CMU dataset — but exact frame rate isn't published and ego-motion is highway-fast.

### Datasets to skip for this use case
- **Boreas, MulRan, Oxford Radar RobotCar, RADIATE** — Navtech scanning radar at 4 Hz; *worse* temporal resolution than our current 5 Hz, and fundamentally different data (mechanical scan vs MIMO).
- **Ithaca-365** — no radar.
- **RadarScenes, CRUW, Radatron, MMVR** — no LiDAR.
- **nuScenes** — single-chip 2D radar, very sparse points.
- **TJ4DRadSet** — LiDAR release pending, restrictive NDA.
- **Zendar** — no longer downloadable.
- **WaveRadar** — could not verify it exists publicly under that name.

---

## Sources

- [ColoRadar paper (arXiv 2103.04510)](https://ar5iv.labs.arxiv.org/html/2103.04510)
- [ColoRadar project page](https://arpg.github.io/coloradar/)
- [K-Radar GitHub](https://github.com/kaist-avelab/K-Radar)
- [K-Radar paper (arXiv 2206.08171)](https://ar5iv.labs.arxiv.org/html/2206.08171)
- [View of Delft documentation](https://tudelft-iv.github.io/view-of-delft-dataset/)
- [VoD paper (IEEE RA-L 2022)](https://ieeexplore.ieee.org/document/9699098/)
- [Boreas project page](https://www.boreas.utias.utoronto.ca/)
- [pyboreas DATA_REFERENCE](https://github.com/utiasASRL/pyboreas/blob/master/DATA_REFERENCE.md)
- [Boreas on AWS Open Data](https://registry.opendata.aws/boreas/)
- [RadarScenes](https://radar-scenes.com/)
- [RadarScenes paper (arXiv 2104.02493)](https://arxiv.org/pdf/2104.02493)
- [MulRan](https://sites.google.com/view/mulran-pr/dataset)
- [MulRan paper (ICRA 2020)](https://gisbi-kim.github.io/publications/gkim-2020-icra.pdf)
- [CRUW project page](https://www.cruwdataset.org/)
- [CRUW paper (arXiv 2105.05207)](https://arxiv.org/abs/2105.05207)
- [aiMotive paper (arXiv 2211.09445)](https://arxiv.org/abs/2211.09445)
- [aiMotive GitHub](https://github.com/aimotive/aimotive_dataset)
- [Astyx HiRes2019 (Papers with Code)](https://paperswithcode.com/dataset/astyx-hires2019)
- [Oxford Radar RobotCar](https://oxford-robotics-institute.github.io/radar-robotcar-dataset/)
- [Oxford Radar RobotCar paper (arXiv 1909.01300)](https://arxiv.org/abs/1909.01300)
- [Ithaca-365](https://ithaca365.mae.cornell.edu/)
- [TJ4DRadSet GitHub](https://github.com/TJRadarLab/TJ4DRadSet)
- [TJ4DRadSet paper (arXiv 2204.13483)](https://arxiv.org/abs/2204.13483)
- [nuScenes paper (arXiv 1903.11027)](https://arxiv.org/abs/1903.11027)
- [RADIal GitHub](https://github.com/valeoai/RADIal)
- [RADIal paper (arXiv 2112.10646)](https://arxiv.org/abs/2112.10646)
- [MSC-RAD4R](https://mscrad4r.github.io/)
- [Snail-Radar project](https://snail-radar.github.io/)
- [Snail-Radar paper (arXiv 2407.11705)](https://arxiv.org/html/2407.11705v2/)
- [Radatron project](https://jguan.page/Radatron/)
- [Radatron ECCV 2022 paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136990157.pdf)
- [Dual Radar (Nature Sci. Data 2025)](https://www.nature.com/articles/s41597-025-04698-2)
- [RadarML project](https://radarml.github.io/)
- [MMVR paper (arXiv 2406.10708)](https://arxiv.org/abs/2406.10708)
- [PixSet (LeddarTech)](https://leddartech.com/solutions/leddar-pixset-dataset/)
- [PixSet paper (arXiv 2102.12010)](https://arxiv.org/abs/2102.12010)
- [RADIATE paper (arXiv 2010.09076)](https://arxiv.org/abs/2010.09076)
- [Awesome Radar Perception list](https://github.com/ZHOUYI1023/awesome-radar-perception)
