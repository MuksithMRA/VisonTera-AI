flowchart TB
    %% ═══════════════════════════════════════════════════════
    %% TRAINING PHASE
    %% ═══════════════════════════════════════════════════════
    subgraph TRAINING["🏋️ MODEL TRAINING PHASE (Offline — One Time)"]
        direction TB

        subgraph DET_TRAIN["🎯 Person Detection Model Training"]
            direction TB
            CH["📦 CrowdHuman Dataset v1 19,369 annotated images Dense crowd scenes"]
            BASE_DET["🧠 YOLOv26m Pre-trained (COCO) General object detection"]
            TL1["⚡ Transfer Learning Crowd-optimized augmentation 100 epochs"]
            CROWD_MODEL["✅ crowd_v_20260219_102246 Person Detection Model Optimized for high-density crowds"]

            CH --> TL1
            BASE_DET --> TL1
            TL1 --> CROWD_MODEL
        end

        subgraph GEN_TRAIN["👤 Gender Classification Model Training"]
            direction TB
            PA["📦 European + KSA Dataset + Custom Web Images 24,382 labeled images"]
            BASE_CLS["🧠 YOLOv26m-cls Pre-trained (ImageNet) Image Classification"]
            TL2["⚡ Transfer Learning Binary classification: Male / Female 80 epochs"]
            GENDER_MODEL["✅ v_20260219_063644 Gender Classification Model 87.35% accuracy"]

            PA --> TL2
            BASE_CLS --> TL2
            TL2 --> GENDER_MODEL
        end
    end

    %% ═══════════════════════════════════════════════════════
    %% MODEL VERSIONING
    %% ═══════════════════════════════════════════════════════
    subgraph VERSIONING["📁 MODEL & DATASET VERSIONING"]
        direction TB

        subgraph MODELS_REPO["🗂️ Model Registry — infrastructure/models/"]
            direction TB
            M1["yolo26m.pt — Base detection model"]
            M2["crowd_v_20260219_102246/ — Crowd person detector (latest)"]
            M3["v_20260219_063644/ — Gender classifier v2 (latest — 87.35%)"]
            M4["v_20260209_170745/ — Gender classifier v1 (archived — 83.55%)"]
        end

        subgraph DATASETS_REPO["🗂️ Dataset Registry — datasets/"]
            direction TB
            DL["📥 to_label/ — Live Collection Auto-collected crops from CCTV Reviewed by human"]
            DN["📂 new_dataset/ — Merged Raw Dataset 24,382 images (current) v_20260209_170745/ — Archived previous version"]
            DY["📂 new_dataset_yolo/ — YOLO-Ready Dataset train/ + val/ (80/20 split) v_20260209_170745/ — Archived previous split"]
        end

        subgraph CROWD_DS["🗂️ Crowd Dataset — research/crowdhuman/"]
            direction TB
            DC["📂 CrowdHuman v1 (Roboflow) train/ valid/ test/ 19,369 images"]
        end

        DL -->|"Human review + merge"| DN
        DN -->|"Auto-convert to YOLO format"| DY
    end

    %% ═══════════════════════════════════════════════════════
    %% REAL-TIME INFERENCE
    %% ═══════════════════════════════════════════════════════
    subgraph INFERENCE["🔴 REAL-TIME INFERENCE PIPELINE (Live — Every Frame)"]
        direction TB

        CAM["📹 CCTV Camera Feed Multiple cameras Real-time video stream"]

        subgraph STAGE1["STAGE 1 — Person Detection (~5ms/frame)"]
            direction TB
            DET["🎯 YOLOv26m Detection crowd_v_20260219_102246"]
            BBOX["📍 Bounding Boxes Detect every person in the frame"]
            DET --> BBOX
        end

        subgraph STAGE2["STAGE 2 — Per-Person Analysis (~1.2ms/person)"]
            direction TB
            CROP["✂️ Crop Each Person"]
            GEN["👤 YOLOv26m-cls v_20260219_063644"]
            RESULT["🏷️ Male / Female + Confidence Score"]
            CROP --> GEN --> RESULT
        end

        subgraph OUTPUT["📊 OUTPUT — Dashboard & Analytics"]
            direction LR
            COUNT["🔢 People Count Total: 47 Male: 28 | Female: 19"]
            LIVE["🖥️ Live Video Annotated feed with bounding boxes + labels"]
            API["🌐 REST API Real-time data to backend systems"]
        end

        CAM --> STAGE1
        STAGE1 --> STAGE2
        STAGE2 --> OUTPUT
    end

    %% ═══════════════════════════════════════════════════════
    %% CONNECTIONS — Training to Inference
    %% ═══════════════════════════════════════════════════════
    CROWD_MODEL -.->|"Deploy"| DET
    GENDER_MODEL -.->|"Deploy"| GEN

    %% Dataset to Training connections
    DC -.->|"Person annotations"| CH
    DY -.->|"YOLO format"| PA

    %% ═══════════════════════════════════════════════════════
    %% STYLING
    %% ═══════════════════════════════════════════════════════
    classDef training fill:#1a1a2e,stroke:#e94560,color:#fff,stroke-width:2px
    classDef detection fill:#0f3460,stroke:#e94560,color:#fff,stroke-width:2px
    classDef gender fill:#533483,stroke:#e94560,color:#fff,stroke-width:2px
    classDef camera fill:#16213e,stroke:#0f3460,color:#fff,stroke-width:2px
    classDef output fill:#1a1a2e,stroke:#e94560,color:#fff,stroke-width:2px
    classDef model fill:#e94560,stroke:#fff,color:#fff,stroke-width:2px,font-weight:bold
    classDef version fill:#0a3d62,stroke:#38ada9,color:#fff,stroke-width:2px

    class CH,PA,BASE_DET,BASE_CLS training
    class DET,BBOX,CROP detection
    class GEN,RESULT gender
    class CAM camera
    class COUNT,LIVE,API output
    class CROWD_MODEL,GENDER_MODEL model
    class M1,M2,M3,M4,DL,DN,DY,DC version
