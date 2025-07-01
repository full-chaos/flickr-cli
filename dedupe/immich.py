try:
    import coremltools as ct
    import coremltools.models.datatypes as dt

    COREML_AVAILABLE = True
except ImportError:
    COREML_AVAILABLE = False
    print("Warning: CoreML not available. Install coremltools for faster inference.")
    
    
@cli.command()
def convert_to_coreml():
    """Convert the ONNX CLIP model to CoreML format for faster inference on macOS."""
    global coreml_model

    if not COREML_AVAILABLE:
        print("Error: coremltools not available. Install it with:")
        print("pip install coremltools")
        return

    import coremltools as ct

    # Try multiple ONNX source locations
    onnx_paths = [
        Path("cache/clip/ViT-B-32__openai/visual/model.onnx"),
        Path("models/clip_vit_b32.onnx"),
        Path("output_onnx_clip/model.onnx"),
    ]

    # Default output path in simplified structure
    coreml_path = Path("cache/clip/ViT-B-32__openai/visual/model.mlmodel")

    # Find available ONNX model
    onnx_path = None
    for path in onnx_paths:
        if path.exists():
            onnx_path = path
            break

    if onnx_path is None:
        print("Error: No ONNX model found in:")
        for path in onnx_paths:
            print(f"  {path}")
        print("Please ensure you have a CLIP ONNX model available.")
        return

    # Create output directory if needed
    coreml_path.parent.mkdir(parents=True, exist_ok=True)

    if coreml_path.exists():
        overwrite = click.confirm(
            f"CoreML model already exists at {coreml_path}. Overwrite?"
        )
        if not overwrite:
            return

    print(f"Converting {onnx_path} to CoreML...")
    print("This may take a few minutes...")

    try:
        # Try basic conversion first
        print("Attempting basic ONNX to CoreML conversion...")
        model = ct.converters.onnx.convert(str(onnx_path))

        # Save the CoreML model
        model.save(str(coreml_path))
        print(f"Successfully converted to CoreML: {coreml_path}")
        print("You can now use --method=coreml for faster inference!")

    except Exception as e:
        print(f"Direct ONNX conversion failed: {e}")
        print("\nTrying alternative approach with PyTorch conversion...")

        try:
            # Alternative: Load with transformers and convert via PyTorch
            from transformers import CLIPVisionModel
            import torch

            # Load the PyTorch model
            model_name = "openai/clip-vit-base-patch16"
            vision_model = CLIPVisionModel.from_pretrained(model_name)
            vision_model.eval()

            # Create dummy input
            dummy_input = torch.randn(1, 3, 224, 224)

            # Convert to CoreML via PyTorch
            traced_model = torch.jit.trace(vision_model, dummy_input)
            coreml_model_converted = ct.convert(
                traced_model, inputs=[ct.TensorType(shape=(1, 3, 224, 224))]
            )

            coreml_model_converted.save(str(coreml_path))
            print(f"Successfully converted via PyTorch: {coreml_path}")

        except Exception as e2:
            print(f"PyTorch conversion also failed: {e2}")
            print("\nPlease check:")
            print("1. CoreML Tools installation: 'pip install --upgrade coremltools'")
            print("2. PyTorch installation: 'pip install torch'")
            print("3. ONNX model validity")
            return

    # Reload the global CoreML model
    try:
        coreml_model = ct.models.MLModel(str(coreml_path))
        print("CoreML model loaded and ready to use!")
    except Exception as e:
        print(f"Warning: Could not load converted model: {e}")

