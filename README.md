# strata
Command-line tool to export PBR textures from PSD layers.

## Overview
`strata` extracts PBR textures from PSD files with named layers, generates normal maps from height maps, and combines texture channels.

## Features
- Extract Albedo/Base Color textures
- Generate Normal Maps from height maps
- Create packed ORM (Occlusion-Roughness-Metallic) textures
- Configure layer names
- Set output texture resolution
- Batch process multiple PSD files with individual settings
- Choose between full-canvas or cropped layer export

### Pre-built Binaries
Binaries for Windows, macOS, and Linux are available on the [Releases](https://github.com/retran/strata/releases) page.

## Usage
Basic usage:
```
strata --input <path_to_PSD> --output <output_folder>
```

Advanced options:
```
strata --input <path_to_PSD> --output <output_folder> --size 2048 --normal-strength 5.0 --export-height --crop-layers
```

## Configuration File
```
strata --config config.json
```

Standard configuration:
```json
{
  "input_file": "input.psd",
  "output_dir": "output/",
  "texture_size": 2048,
  "normal_strength": 5.0,
  "export_heightmap": true,
  "crop_layers": false,
  "layer_names": {
    "albedo": "basecolor",
    "heightmap": "displacement",
    "occlusion": "ambientocclusion",
    "roughness": "specular",
    "metallic": "metal"
  }
}
```

## Batch Processing
Process multiple files with individual settings:

```json
{
  "output_dir": "output/",
  "texture_size": 2048,
  "layer_names": {
    "albedo": "basecolor",
    "heightmap": "displacement",
    "occlusion": "ambientocclusion",
    "roughness": "specular",
    "metallic": "metal"
  },
  "input_files": [
    {
      "input_file": "texture1.psd",
      "texture_size": 4096,
      "normal_strength": 3.0
    },
    {
      "input_file": "texture2.psd",
      "output_dir": "output/special/",
      "crop_layers": true,
      "layer_names": {
        "albedo": "diffuse",
        "roughness": "glossiness"
      }
    },
    {
      "input_file": "texture3.psd"
    }
  ]
}
```

Batch processing supports:
- Per-file parameter overrides
- Global defaults for non-specified parameters
- All parameters are optional

## Configuration Options
- `input_file`: Path to the input PSD file
- `input_files`: Array of files for batch processing
- `output_dir`: Directory for exported textures
- `texture_size`: Output resolution (default: 1024)
- `normal_strength`: Normal map intensity (default: 4.0)
- `export_heightmap`: Whether to export the heightmap (default: false)
- `crop_layers`: Export only layer content without positioning on full canvas (default: false)
- `layer_names`: Layer name mappings

## Default Layer Names
- `albedo`: Base color texture
- `heightmap`: Height/displacement map
- `occlusion`: Ambient occlusion map
- `roughness`: Surface roughness map
- `metallic`: Metallic map

## Command-line Options
- `--input`: Path to the input PSD file
- `--output`: Directory for exported textures
- `--config`: Path to JSON config file
- `--size`, `-s`: Target size for output textures (default: 1024)
- `--normal-strength`, `-n`: Normal map strength (default: 4.0)
- `--export-height`: Export the heightmap (disabled by default)
- `--crop-layers`: Export only layer content without positioning on full canvas (default: full-size export)
- `--verbose`, `-v`: Enable verbose logging
- `--albedo-layer`: Custom name for albedo layer (default: "albedo")
- `--heightmap-layer`: Custom name for heightmap layer (default: "heightmap")
- `--occlusion-layer`: Custom name for occlusion layer (default: "occlusion")
- `--roughness-layer`: Custom name for roughness layer (default: "roughness")
- `--metallic-layer`: Custom name for metallic layer (default: "metallic")

## Output Files
- `<filename>_albedo.png` - RGB albedo/base color texture
- `<filename>_heightmap.png` - Grayscale height map (optional)
- `<filename>_normal.png` - RGB normal map
- `<filename>_orm.png` - Packed texture (R: Occlusion, G: Roughness, B: Metallic)

## Layer Positioning Modes
By default, Strata preserves the original positioning of layers on the full PSD canvas. This ensures textures maintain proper alignment with each other:

- **Full-size Export (default)**: Preserves layer positions on the PSD canvas
- **Crop Layers**: Only exports the actual content of each layer, ignoring their positions

Use the `--crop-layers` flag or set `"crop_layers": true` in your config to crop layers.

## Build
Build executable:
- macOS: `./build/build_macos.sh`
- Linux: `./build/build_linux.sh`
- Windows: `build\build_windows.cmd`

## Version History
### v0.3.0 (April 2025)
- Added crop layers option to export only layer content without positioning
- Improved CLI interface with named arguments
- Fixed color profile handling in albedo exports
- Added direct layer name customization via CLI arguments

### v0.2.0
- Added batch processing for multiple PSD files
- Added per-file configuration
- Made all configuration parameters optional
- Improved error handling

### v0.1.0
- Initial release
