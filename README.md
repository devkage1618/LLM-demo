# Panorama to Perspective Video Converter


## Requirements

Make sure you have Python 3 and the following libraries installed:

```bash
pip install numpy opencv-python scipy av opencv-contrib-python --only-binary :all:
```

## Command
```bash
python calib.py
python main.py <input_video> <output_video>
```

## Example

```bash
python calib.py
```

- Click "Open" Button and load a video "input/clip2.mp4".
- Left column sliders are for camera config settings.
- Everytime you changed a slider, you can see the result by clicking "Preview" Button. (After once focused, "Space" Keyboard)
- Click "Save" Button (It will make a config file in the same directory)

```bash
python main.py input/clip2.mp4 output/result2.mp4
```
It will load clip2_config.txt and clip2_coordinates.txt then process video saved filename is "result2.mp4"