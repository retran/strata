#!/usr/bin/env python3
"""
Strata - PBR Texture Exporter

Extracts PBR (Physically Based Rendering) textures from PSD files with named layers.
Generates normal maps from height maps and combines AO/roughness/metallic into ORM textures.
"""

import os
import sys
import io
import logging
import argparse
from pathlib import Path
from psd_tools import PSDImage
from PIL import Image, ImageCms
import traceback
import numpy as np
from scipy import ndimage
import json


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger('texture_exporter')


def create_normal_map(height_image_pil, strength):
    """
    Generates a normal map from a height map.

    Args:
        height_image_pil: Height map PIL Image (grayscale preferred)
        strength: Controls bump intensity (higher = steeper slopes)

    Returns:
        PIL Image: RGB normal map
    """
    logger.info(f"Generating Normal Map (Strength: {strength})...")
    # Convert to grayscale and normalize to 0-1 range
    height_map = np.array(height_image_pil.convert('L')).astype(float) / 255.0

    # Calculate gradients with Sobel filters
    dz_dx = ndimage.sobel(height_map, axis=1)  # X gradient
    dz_dy = ndimage.sobel(height_map, axis=0)  # Y gradient

    # Create normal vector components (OpenGL convention)
    normal_x = -dz_dx * strength
    normal_y = -dz_dy * strength
    normal_z = np.ones_like(height_map)

    # Normalize vectors to unit length
    magnitude = np.sqrt(normal_x**2 + normal_y**2 + normal_z**2)
    # Prevent division by zero
    magnitude[magnitude == 0] = 1e-6
    normal_x /= magnitude
    normal_y /= magnitude
    normal_z /= magnitude

    # Map from [-1, 1] to [0, 255] for RGB
    normal_map_rgb = np.zeros((height_map.shape[0], height_map.shape[1], 3), dtype=np.uint8)
    normal_map_rgb[..., 0] = (normal_x * 0.5 + 0.5) * 255  # R <- X
    normal_map_rgb[..., 1] = (normal_y * 0.5 + 0.5) * 255  # G <- Y
    normal_map_rgb[..., 2] = (normal_z * 0.5 + 0.5) * 255  # B <- Z

    # Convert back to PIL Image
    normal_image_pil = Image.fromarray(normal_map_rgb, 'RGB')
    logger.info("Normal Map generation complete.")
    return normal_image_pil


def convert_to_srgb(img):
    """
    Convert image to sRGB color space

    Args:
        img: Input PIL Image

    Returns:
        PIL Image: sRGB converted image
    """
    temp_img = img  # Work on a copy
    # Convert to RGB first if needed
    if temp_img.mode not in ['RGB', 'RGBA']:
        temp_img = temp_img.convert('RGBA' if 'A' in img.mode else 'RGB')

    try:
        # If image has profile, convert from that to sRGB
        if hasattr(temp_img, 'info') and 'icc_profile' in temp_img.info and temp_img.info['icc_profile']:
            logger.info("Converting from embedded profile to sRGB...")
            input_profile = ImageCms.ImageCmsProfile(io.BytesIO(temp_img.info['icc_profile']))
            srgb_profile = ImageCms.createProfile('sRGB')
            # Ensure mode is compatible with profile conversion
            if temp_img.mode != 'RGB':
                temp_img = temp_img.convert('RGB')
            return ImageCms.profileToProfile(temp_img, input_profile, srgb_profile, outputMode='RGB')
        # Otherwise assume linear/undefined
        else:
            logger.info("No embedded profile found. Using default sRGB conversion.")
            if temp_img.mode != 'RGB':
                temp_img = temp_img.convert('RGB')
            return temp_img
    except Exception as e:
        logger.warning(f"Color profile conversion failed: {e}")
        if temp_img.mode != 'RGB':
            temp_img = temp_img.convert('RGB')
        return temp_img


def position_layer_on_full_canvas(layer, psd_size):
    """
    Place a layer on a full-size canvas preserving its original position

    Args:
        layer: PSD layer object
        psd_size: Tuple (width, height) of the full PSD document

    Returns:
        PIL Image: Layer positioned on a full-size canvas
    """
    if not layer or not layer.has_pixels():
        return None

    # Get the layer's PIL image
    layer_img = layer.topil()

    # Create a transparent canvas the size of the full PSD
    mode = 'RGBA' if 'A' in layer_img.mode else 'RGB'
    bg_color = (0, 0, 0, 0) if mode == 'RGBA' else (0, 0, 0)
    canvas = Image.new(mode, psd_size, color=bg_color)

    # Get layer's position
    x, y = layer.offset

    # Paste the layer onto the canvas at its original position
    if mode == 'RGBA':
        canvas.paste(layer_img, (x, y), layer_img if 'A' in layer_img.mode else None)
    else:
        canvas.paste(layer_img, (x, y))

    logger.info(f"Positioned layer on full canvas ({psd_size[0]}x{psd_size[1]}) at offset ({x}, {y})")
    return canvas


def resize_image(img, target_size):
    """
    Resize image to target size

    Args:
        img: Input PIL Image
        target_size: Tuple (width, height) for output

    Returns:
        PIL Image: Resized image
    """
    if img.size != target_size:
        logger.info(f"Resizing from {img.size} to {target_size} using LANCZOS...")
        # Use Resampling enum for newer Pillow versions
        resample_filter = Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS
        return img.resize(target_size, resample_filter)
    return img


def save_image(img, path, format='PNG'):
    """
    Save image to disk

    Args:
        img: PIL Image to save
        path: Path where to save the image
        format: Image format (default: PNG)

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        img.save(path, format=format)
        logger.info(f"Saved: {path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save image to {path}: {e}")
        return False


def get_layer_from_psd(psd, layer_names):
    """
    Search for specific layers in a PSD file by name

    Args:
        psd: PSDImage object
        layer_names: List of layer names to search for

    Returns:
        dict: Dictionary of found layers and count
    """
    found_layers = {name: None for name in layer_names}
    found_count = 0

    logger.info("Searching for required layers...")
    # Search recursively through all layers and groups
    for layer in psd.descendants():
        # Skip layers without pixels
        if not layer.has_pixels():
            continue

        layer_name_lower = layer.name.strip().lower()

        if layer_name_lower in found_layers:
            if found_layers[layer_name_lower] is None:  # Take the first one found
                logger.info(f"Found layer: '{layer.name}' -> as '{layer_name_lower}' (Size: {layer.width}x{layer.height})")
                found_layers[layer_name_lower] = layer
                found_count += 1
            else:
                # Warning for duplicate layers
                logger.warning(f"Found duplicate layer for '{layer_name_lower}': '{layer.name}'. Using the first one found.")

    logger.info(f"Search completed. Found {found_count} out of {len(found_layers)} target layers.")
    return found_layers, found_count


def process_albedo(layer, output_dir, base_filename, target_size, full_size_export=False, psd_size=None):
    """Process and export albedo texture"""
    if not layer:
        logger.info("Layer 'albedo' not found, export skipped.")
        return False

    try:
        logger.info("Exporting Albedo...")

        if full_size_export and psd_size:
            logger.info(f"Using full canvas size: {psd_size[0]}x{psd_size[1]}")
            albedo_img = position_layer_on_full_canvas(layer, psd_size)
            # Then resize to target size if needed
            albedo_img = resize_image(albedo_img, target_size)
        else:
            # Original method
            albedo_img = layer.topil()
            albedo_img = resize_image(albedo_img, target_size)

        albedo_img = convert_to_srgb(albedo_img)  # Handle color space

        # Convert to RGB if needed
        if albedo_img.mode != 'RGB':
            logger.warning(f"Albedo not in RGB mode after conversion attempt ({albedo_img.mode}). Converting.")
            albedo_img = albedo_img.convert('RGB')

        output_path = os.path.join(output_dir, f"{base_filename}_albedo.png")
        return save_image(albedo_img, output_path)
    except Exception as e:
        logger.error(f"Error when exporting Albedo: {e}")
        traceback.print_exc()
        return False


def process_height_and_normal(layer, output_dir, base_filename, target_size, normal_strength, export_height=True, full_size_export=False, psd_size=None):
    """Process heightmap and generate normal map"""
    if not layer:
        logger.info("Layer 'heightmap' not found, Normal export skipped.")
        return False

    try:
        logger.info("Processing Heightmap...")

        if full_size_export and psd_size:
            logger.info(f"Using full canvas size: {psd_size[0]}x{psd_size[1]}")
            height_img = position_layer_on_full_canvas(layer, psd_size)
            # Resize to target size if needed
            height_img_resized = resize_image(height_img, target_size)
        else:
            # Original method
            height_img = layer.topil()  # Get PIL Image
            height_img_resized = resize_image(height_img, target_size)

        # Export heightmap if requested
        if export_height:
            height_output_path = os.path.join(output_dir, f"{base_filename}_heightmap.png")
            # Convert to grayscale (8-bit) for export
            height_img_gray = height_img_resized.convert('L')
            save_image(height_img_gray, height_output_path)

        # Generate and Export Normal Map
        logger.info("Exporting Normal map...")
        normal_img = create_normal_map(height_img_resized, strength=normal_strength)
        normal_output_path = os.path.join(output_dir, f"{base_filename}_normal.png")
        return save_image(normal_img, normal_output_path)
    except Exception as e:
        logger.error(f"Error when processing Heightmap/Normal map: {e}")
        traceback.print_exc()
        return False


def process_orm(ao_layer, roughness_layer, metallic_layer, output_dir, base_filename, target_size, full_size_export=False, psd_size=None):
    """Process and export ORM (Occlusion, Roughness, Metallic) texture"""
    # Check if all three layers exist
    if not (ao_layer and roughness_layer and metallic_layer):
        logger.info("Skipping ORM texture creation as one or more layers were not found:")
        if not ao_layer: logger.info("- Layer 'occlusion' is missing.")
        if not roughness_layer: logger.info("- Layer 'roughness' is missing.")
        if not metallic_layer: logger.info("- Layer 'metallic' is missing.")
        return False

    try:
        logger.info("Creating ORM texture (R=Occlusion, G=Roughness, B=Metallic)...")

        if full_size_export and psd_size:
            logger.info(f"Using full canvas size: {psd_size[0]}x{psd_size[1]} for ORM layers")
            # Position each layer on a full-size canvas
            ao_pil = position_layer_on_full_canvas(ao_layer, psd_size)
            roughness_pil = position_layer_on_full_canvas(roughness_layer, psd_size)
            metallic_pil = position_layer_on_full_canvas(metallic_layer, psd_size)
        else:
            # Original method - get images directly
            ao_pil = ao_layer.topil()
            roughness_pil = roughness_layer.topil()
            metallic_pil = metallic_layer.topil()

        # Check if image sizes match BEFORE resizing originals
        if not (ao_pil.size == roughness_pil.size == metallic_pil.size):
            logger.warning(f"Original sizes of Occlusion ({ao_pil.size}), Roughness ({roughness_pil.size}) and Metallic ({metallic_pil.size}) layers don't match. Resizing each to target size before merge.")

        # Convert to grayscale and resize to target size
        ao_img = resize_image(ao_pil.convert('L'), target_size)
        roughness_img = resize_image(roughness_pil.convert('L'), target_size)
        metallic_img = resize_image(metallic_pil.convert('L'), target_size)

        # Double check sizes after potential resizing
        if not (ao_img.size == roughness_img.size == metallic_img.size == target_size):
            logger.error(f"Sizes after resize don't match target or each other. Occlusion: {ao_img.size}, Rough: {roughness_img.size}, Metal: {metallic_img.size}. Cannot create ORM.")
            return False

        logger.info(f"Merging channels (Size: {ao_img.size})...")
        # Create a new RGB image using channels from Occlusion, Roughness, Metallic
        orm_image = Image.merge('RGB', (ao_img, roughness_img, metallic_img))

        output_path = os.path.join(output_dir, f"{base_filename}_orm.png")
        return save_image(orm_image, output_path)
    except Exception as e:
        logger.error(f"Error when creating ORM texture: {e}")
        traceback.print_exc()
        return False


def export_pbr_textures(psd_filepath, output_dir, target_size=None, normal_strength=4.0,
                       export_height=False, verbose=False, layer_names=None, full_size_export=True):
    """
    Exports PBR textures (Albedo, Heightmap, Normal, ORM) from PSD file layers,
    using psd-tools and Pillow. Generates Normal map from Heightmap.

    Args:
        psd_filepath: Path to the PSD file
        output_dir: Directory to save exported textures
        target_size: Tuple (width, height) for output textures (default: same as PSD dimensions)
        normal_strength: Strength factor for normal map generation (default: 4.0)
        export_height: Whether to export the heightmap (default: False)
        verbose: Enable verbose logginsg (default: False)
        layer_names: Dictionary mapping texture types to layer names (default: standard names)
        full_size_export: Whether to export layers positioned on the full PSD canvas (default: True)

    Returns:
        dict: Status of each exported texture
    """
    # Set log level based on verbose flag
    if verbose:
        logger.setLevel(logging.DEBUG)

    logger.info("--- Starting processing ---")
    logger.info(f"PSD file: {psd_filepath}")
    logger.info(f"Output folder: {output_dir}")

    # Default target size to PSD dimensions if not provided
    psd = PSDImage.open(psd_filepath)
    if target_size is None:
        target_size = (psd.width, psd.height)
    logger.info(f"Output size: {target_size[0]}x{target_size[1]}")
    logger.info(f"Normal map strength: {normal_strength}")
    logger.info(f"Export heightmap: {export_height}")

    # Default layer names if not provided
    default_layer_names = {
        "albedo": "albedo",
        "occlusion": "occlusion",
        "metallic": "metallic",
        "heightmap": "heightmap",
        "roughness": "roughness"
    }

    # Use provided layer names or defaults
    target_layer_names = layer_names or default_layer_names
    logger.info(f"Using layer names: {target_layer_names}")

    # Track success of each export operation
    export_status = {
        "albedo": False,
        "heightmap": False,
        "normal": False,
        "orm": False
    }

    # --- Checks and preparation ---
    if not os.path.exists(psd_filepath):
        logger.error(f"PSD file not found: {psd_filepath}")
        return export_status

    if not os.path.isfile(psd_filepath):
        logger.error(f"Specified path is not a file: {psd_filepath}")
        return export_status

    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create output folder '{output_dir}': {e}")
        return export_status

    base_filename = os.path.splitext(os.path.basename(psd_filepath))[0]
    logger.info(f"Base filename: {base_filename}")

    # --- Loading PSD and searching for layers ---
    psd = None
    try:
        logger.info("Loading PSD (may take time)...")
        psd = PSDImage.open(psd_filepath)
        target_layers, _ = get_layer_from_psd(psd, list(target_layer_names.values()))
    except Exception as e:
        logger.error(f"Failed to load or analyze PSD file: {e}")
        traceback.print_exc()
        return export_status

    # --- Process and export textures ---
    # Albedo
    export_status["albedo"] = process_albedo(
        target_layers.get(target_layer_names["albedo"]), output_dir, base_filename, target_size, full_size_export, psd.size
    )

    # Heightmap and Normal
    height_normal_result = process_height_and_normal(
        target_layers.get(target_layer_names["heightmap"]), output_dir, base_filename,
        target_size, normal_strength, export_height, full_size_export, psd.size
    )
    export_status["normal"] = height_normal_result
    export_status["heightmap"] = height_normal_result and export_height

    # ORM (Occlusion, Roughness, Metallic)
    export_status["orm"] = process_orm(
        target_layers.get(target_layer_names["occlusion"]),
        target_layers.get(target_layer_names["roughness"]),
        target_layers.get(target_layer_names["metallic"]),
        output_dir, base_filename, target_size, full_size_export, psd.size
    )

    # --- Final status ---
    logger.info("--- Processing completed ---")
    logger.info("Export status:")
    for texture, status in export_status.items():
        logger.info(f"  {texture.capitalize()}: {'Success' if status else 'Failed/Skipped'}")

    return export_status


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Export PBR textures from PSD file layers",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # Create a mutually exclusive group for the two modes of operation
    input_group = parser.add_mutually_exclusive_group(required=True)

    # Add --config as one option in the group
    input_group.add_argument("--config", help="Path to JSON config file with processing settings")

    # Add the file input arguments as another option in the group
    input_group.add_argument("--input", dest="psd_file",
                          help="Path to the PSD file (not required when using --config)")
    parser.add_argument("--output", dest="output_dir",
                      help="Directory to save exported textures (not required when using --config)")

    # Keep the rest of the arguments unchanged
    parser.add_argument(
        "--size", "-s", type=int, default=1024,
        help="Target size for output textures (default: 1024)"
    )
    parser.add_argument(
        "--normal-strength", "-n", type=float, default=4.0,
        help="Strength factor for normal map generation (default: 4.0)"
    )
    parser.add_argument(
        "--export-height", action="store_true",
        help="Export the heightmap (disabled by default)"
    )
    parser.add_argument(
        "--crop-layers", action="store_true",
        help="Export only the layer content without positioning on full canvas (disables default full-size export)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging"
    )

    # Layer name configuration options
    layer_group = parser.add_argument_group('Layer Names', 'Customize the PSD layer names to search for')
    layer_group.add_argument("--albedo-layer", default="albedo", help="Layer name for albedo/base color (default: albedo)")
    layer_group.add_argument("--heightmap-layer", default="heightmap", help="Layer name for heightmap/displacement (default: heightmap)")
    layer_group.add_argument("--occlusion-layer", default="occlusion", help="Layer name for ambient occlusion (default: occlusion)")
    layer_group.add_argument("--roughness-layer", default="roughness", help="Layer name for roughness (default: roughness)")
    layer_group.add_argument("--metallic-layer", default="metallic", help="Layer name for metallic (default: metallic)")

    args = parser.parse_args()

    # Validate that output_dir is provided when using --input
    if args.psd_file and not args.output_dir:
        parser.error("--output is required when using --input")

    return args


# --- Main execution block ---
def main():
    """Main entry point for the application"""
    # Handle no arguments case first
    if len(sys.argv) <= 1:
        print("Usage: strata --input <psd_file> --output <output_dir> [options]")
        print("   OR: strata --config <config_file>")
        print("\nFor more information, use --help")
        sys.exit(1)

    # For help flag, let argparse handle it
    if len(sys.argv) == 2 and sys.argv[1] in ['-h', '--help']:
        parse_arguments()  # This will print help and exit
        sys.exit(0)

    try:
        # Parse arguments (new argparse-based CLI)
        args = parse_arguments()

        # Default settings from command line arguments
        input_file = args.psd_file
        output_dir = args.output_dir
        texture_size = args.size
        normal_strength = args.normal_strength
        export_height = args.export_height

        # Handle layer name configuration
        layer_names = {
            "albedo": args.albedo_layer,
            "heightmap": args.heightmap_layer,
            "occlusion": args.occlusion_layer,
            "roughness": args.roughness_layer,
            "metallic": args.metallic_layer
        }

        # Config file and batch processing
        config_data = {}
        input_files = []
        input_file_configs = []

        # Load config from file if specified (overrides command line options)
        if args.config:
            try:
                with open(args.config, 'r') as f:
                    config_data = json.load(f)

                    # Update PSD file path if provided in config (for single file mode)
                    if "input_file" in config_data:
                        input_file = config_data["input_file"]
                        logger.info(f"Using input file from config: {input_file}")

                    # Update output directory if provided in config
                    if "output_dir" in config_data:
                        output_dir = config_data["output_dir"]
                        logger.info(f"Using output directory from config: {output_dir}")

                    # Update texture size from config if provided
                    if "texture_size" in config_data:
                        texture_size = config_data["texture_size"]
                        logger.info(f"Using texture size from config: {texture_size}")

                    # Update normal strength from config if provided
                    if "normal_strength" in config_data:
                        normal_strength = config_data["normal_strength"]
                        logger.info(f"Using normal strength from config: {normal_strength}")

                    # Update whether to export height map from config if provided
                    if "export_heightmap" in config_data:
                        export_height = config_data["export_heightmap"]
                        logger.info(f"Export heightmap setting from config: {export_height}")
                    elif "skip_height" in config_data:
                        # For backward compatibility
                        export_height = not config_data["skip_height"]
                        logger.info(f"Export heightmap setting from legacy config option 'skip_height': {export_height}")

                    # Update layer names from config file if provided
                    if "layer_names" in config_data:
                        logger.info(f"Loading layer names from config file: {args.config}")
                        layer_names.update(config_data["layer_names"])

                    # Support legacy psd_file config option for backward compatibility
                    if "psd_file" in config_data:
                        input_file = config_data["psd_file"]
                        logger.info(f"Using input file from legacy config option 'psd_file': {input_file}")

                    # Get list of input files if provided for batch processing
                    if "input_files" in config_data:
                        if isinstance(config_data["input_files"], list):
                            # Check if it's a simple list of strings or a list of objects with configurations
                            if all(isinstance(item, str) for item in config_data["input_files"]):
                                # Simple list of file paths
                                input_files = config_data["input_files"]
                                logger.info(f"Found {len(input_files)} files for batch processing in config")
                                # Each file will use the global settings
                                input_file_configs = [{}] * len(input_files)
                            elif all(isinstance(item, dict) for item in config_data["input_files"]):
                                # List of objects with per-file configurations
                                for item in config_data["input_files"]:
                                    if "input_file" in item:
                                        input_files.append(item["input_file"])
                                        input_file_configs.append(item)
                                    else:
                                        logger.warning(f"Skipping batch item missing 'input_file' field: {item}")
                                logger.info(f"Found {len(input_files)} files with individual configurations for batch processing")
                            else:
                                logger.warning("'input_files' in config contains mixed types. Expected all strings or all objects.")
                        else:
                            logger.warning("'input_files' in config is not a list. Ignoring.")

                    # Validate that we have input and output defined for single file mode
                    if not input_files and not input_file:
                        logger.error("No input file specified in config. Please provide 'input_file' or 'input_files'.")
                        sys.exit(1)

                    if not input_files and not output_dir:
                        logger.error("No output directory specified in config. Please provide 'output_dir'.")
                        sys.exit(1)

            except Exception as e:
                logger.error(f"Failed to load config file {args.config}: {e}")
                sys.exit(1)

        # Process files based on configuration
        if input_files:
            # Batch processing mode
            logger.info(f"Starting batch processing of {len(input_files)} files...")
            batch_results = {}

            for idx, batch_file in enumerate(input_files):
                logger.info(f"\n=== Processing file: {batch_file} ===")

                # Get per-file configuration (if available)
                file_config = input_file_configs[idx]

                # Apply per-file parameters if specified (otherwise use global settings)
                file_texture_size = file_config.get("texture_size", texture_size)
                file_normal_strength = file_config.get("normal_strength", normal_strength)
                file_export_height = file_config.get("export_heightmap", export_height)
                file_output_dir = file_config.get("output_dir", output_dir)

                # Per-file layer name overrides
                file_layer_names = layer_names.copy()
                if "layer_names" in file_config:
                    file_layer_names.update(file_config["layer_names"])
                    logger.info(f"Using custom layer names for this file")

                # Process with potentially customized settings
                logger.info(f"Using texture size: {file_texture_size}, normal strength: {file_normal_strength}")
                if file_output_dir != output_dir:
                    logger.info(f"Using custom output directory: {file_output_dir}")

                # Check for full-size export setting in per-file config
                file_full_size = not args.crop_layers
                if "crop_layers" in file_config:
                    file_full_size = not file_config.get("crop_layers")
                elif "full_size_export" in file_config:
                    file_full_size = file_config.get("full_size_export")

                batch_results[batch_file] = export_pbr_textures(
                    batch_file,
                    file_output_dir,
                    target_size=(file_texture_size, file_texture_size),
                    normal_strength=file_normal_strength,
                    export_height=file_export_height,
                    verbose=args.verbose,
                    layer_names=file_layer_names,
                    full_size_export=file_full_size
                )

            # Summary of batch results
            logger.info("\n=== Batch Processing Summary ===")
            for file_path, file_results in batch_results.items():
                logger.info(f"File: {file_path}")
                for texture, status in file_results.items():
                    logger.info(f"  {texture.capitalize()}: {'Success' if status else 'Failed/Skipped'}")
        else:
            # Single file processing (original behavior)
            export_pbr_textures(
                input_file,
                output_dir,
                target_size=(texture_size, texture_size),
                normal_strength=normal_strength,
                export_height=export_height,
                verbose=args.verbose,
                layer_names=layer_names,
                full_size_export=not args.crop_layers
            )
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()