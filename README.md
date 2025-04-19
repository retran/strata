# strata
Command-line tool to export PBR textures (Albedo, Normal, ORM...) from named PSD layers.

## Overview
`strata` extracts PBR (Physically Based Rendering) textures from PSD files with named layers. It exports standard PBR texture maps and generates normal maps from height maps.

## Features
- Extract Albedo/Base Color textures
- Generate Normal Maps from height maps
- Create packed ORM (Occlusion-Roughness-Metallic) textures
- Configure layer names
- Output textures in consistent sizes

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

### Configuration File
Configure `strata` using a JSON configuration file:

```
strata input.psd output/ --config config.json
```

You can also specify just the config file and have all settings including PSD file path and output directory configured in it:

```
strata --config config.json
```

Sample configuration structure (`examples/config.json`):

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

Configuration options:
- `input_file`: Path to the input PSD file (overrides command-line argument)
- `output_dir`: Directory to save exported textures (overrides command-line argument)
- `texture_size`: Output size for all textures (default: 1024)
- `normal_strength`: Strength of the normal map effect (default: 4.0)
- `export_heightmap`: Whether to export the heightmap (default: true)
- `layer_names`: Custom PSD layer name mappings

Settings in the configuration file override command-line arguments.

### Layer Names Configuration
Default layer names:
- `albedo` - Base color/diffuse texture
- `heightmap` - Height/displacement map (for normal map generation)
- `occlusion` - Ambient occlusion map
- `roughness` - Surface roughness map
- `metallic` - Metallic/metal map

Customize layer names via:

1. Command-line:
```
strata input.psd output/ --albedo-layer basecolor --heightmap-layer displacement --occlusion-layer ambient
```

2. JSON Configuration File (as shown above)

## Build from Source

### Local Build
Build Strata as a standalone executable:

- macOS: `./build/build_macos.sh`
- Linux: `./build/build_linux.sh`
- Windows: `build\build_windows.cmd`

The executable will be in the `dist` folder.

## Options
- `--size`, `-s`: Target size for output textures (default: 1024)
- `--normal-strength`, `-n`: Strength factor for normal map generation (default: 4.0)
- `--skip-height`: Skip exporting the heightmap
- `--verbose`, `-v`: Enable verbose logging
- `--config`: Path to JSON config file with settings

## Output
Generated files:
- `<filename>_albedo.png` - RGB albedo/base color texture
- `<filename>_heightmap.png` - Grayscale height map
- `<filename>_normal.png` - RGB normal map
- `<filename>_orm.png` - Packed texture (R: Occlusion, G: Roughness, B: Metallic)
