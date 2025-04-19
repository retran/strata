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

### Pre-built Binaries
Binaries for Windows, macOS, and Linux are available on the [Releases](https://github.com/retran/strata/releases) page.

## Usage
Basic usage:
```
strata <path_to_PSD> <output_folder>
```

Advanced options:
```
strata <path_to_PSD> <output_folder> --size 2048 --normal-strength 5.0 --skip-height
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
  "layer_names": {
    "albedo": "basecolor",
    "heightmap": "displacement",
    "occlusion": "ambientocclusion",
    "roughness": "specular",
    "metallic": "metal"
  }
}
```

## Batch Processing (v0.2.0)
Process multiple files with individual settings:

```json
{
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
- `export_heightmap`: Whether to export the heightmap (default: true)
- `layer_names`: Layer name mappings

## Default Layer Names
- `albedo`: Base color texture
- `heightmap`: Height/displacement map
- `occlusion`: Ambient occlusion map
- `roughness`: Surface roughness map
- `metallic`: Metallic map

## Command-line Options
- `--size`, `-s`: Target size for output textures (default: 1024)
- `--normal-strength`, `-n`: Normal map strength (default: 4.0)
- `--skip-height`: Skip exporting the heightmap
- `--verbose`, `-v`: Enable verbose logging
- `--config`: Path to JSON config file

## Output Files
- `<filename>_albedo.png` - RGB albedo/base color texture
- `<filename>_heightmap.png` - Grayscale height map
- `<filename>_normal.png` - RGB normal map
- `<filename>_orm.png` - Packed texture (R: Occlusion, G: Roughness, B: Metallic)

## Build
Build executable:
- macOS: `./build/build_macos.sh`
- Linux: `./build/build_linux.sh`
- Windows: `build\build_windows.cmd`

## Version History
### v0.2.0 (April 2025)
- Added batch processing for multiple PSD files
- Added per-file configuration
- Made all configuration parameters optional
- Improved error handling

### v0.1.0
- Initial release
