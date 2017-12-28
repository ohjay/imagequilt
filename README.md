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
python main.py --texture <texture_path> --target <target_path> --patch_height <int> --patch_width <int> --overlap_height <int> --overlap_width <int> --err_threshold <float> --n <int> --outdir <str>
```

In all of the above commands, any omitted parameters will adhere to the following defaults:
```python
out_height     = max(576, image_height * 2)
out_width      = max(576, image width  * 2)
patch_height   = min(32,  image_height * 0.25)
patch_width    = min(32,  image_width  * 0.25)
overlap_height = min(16,  patch_height * 0.5)
overlap_width  = min(16,  patch_width  * 0.5)
err_threshold  = 0.05
n              = 8
outdir         = 'out'
```
