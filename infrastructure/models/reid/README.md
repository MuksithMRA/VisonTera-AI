# Person Re-ID weights (OSNet)

Place **OSNet x1.0** trained on **Market-1501** here as:

`osnet_x1_0_market1501.pth`

Official checkpoint (torchreid model zoo):  
[Google Drive – osnet_x1_0 Market1501](https://drive.google.com/file/d/1vduhq5DpN2q1g4fYEZfPI17MJeh9qyrA/view)

With `gdown` installed, the app can auto-download on first run if the file is missing (`REID_AUTO_DOWNLOAD=true`).

Override path: environment variable `REID_WEIGHTS_PATH` or `AppConfig.REID_WEIGHTS_PATH`.

Legacy FastReID-style ResNet checkpoints (`*reid*.pt`, `*sbs*.pt`) in this folder are still supported as a fallback.
