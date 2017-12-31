# `imagequilt`
Reimplementation of [_Image Quilting for Texture Synthesis and Transfer_](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/papers/efros-siggraph01.pdf) (by Efros and Freeman).

## Usage
#### Synthesizing texture
```
python main.py --texture <texture_path>
python main.py --texture <texture_path> --outsize <int> --patchsize <int> --overlap <int>
python main.py --texture <texture_path> --out_height <int> --out_width <int> --patch_height <int> --patch_width <int> --overlap_height <int> --overlap_width <int> --err_threshold <float> --outdir <str>
```

#### Transferring texture
```
python main.py --texture <texture_path> --target <target_path>
python main.py --texture <texture_path> --target <target_path> --patchsize <int> --overlap <int>
python main.py --texture <texture_path> --target <target_path> --patch_height <int> --patch_width <int> --overlap_height <int> --overlap_width <int> --err_threshold <float> --alpha_init <float> --n <int> --outdir <str>
```

In all of the above commands, any omitted parameters will adhere to the following defaults:
```python
out_height     = max(image_height, image_width) * 3
out_width      = max(image_height, image_width) * 3
patch_height   = max(1, min(image_height, image_width) // 3)
patch_width    = max(1, min(image_height, image_width) // 3)
overlap_height = max(1, patch_height // 3)
overlap_width  = max(1, patch_width  // 3)
err_threshold  = 0.15 if synthesis, 0.05 if transfer
alpha_init     = 0.1
n              = 8
outdir         = 'out'
```
