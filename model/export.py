"""
Export trained OCR model to ONNX.
===================================
Uses fast_plate_ocr's own internal loader.
Auto-patches Erfc nodes after export.

Usage:
    python model/export.py
"""

import tempfile
import pathlib
import sys
import logging

# Force temp files to project directory
ROOT = pathlib.Path(__file__).parent.parent
TMP_DIR = ROOT / "tmp"
TMP_DIR.mkdir(exist_ok=True)

_ORIG_NTF = tempfile.NamedTemporaryFile
def _patched_ntf(*args, **kwargs):
    kwargs.setdefault("dir", str(TMP_DIR))
    kwargs.setdefault("delete", False)
    return _ORIG_NTF(*args, **kwargs)
tempfile.NamedTemporaryFile = _patched_ntf

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════

KERAS_MODEL   = ROOT / "model" / "output" / "2026-05-19_11-51-38" / "best.keras"
PLATE_CONFIG  = ROOT / "config" / "plate_config.yaml"
SAVE_DIR      = ROOT / "model" / "output" / "2026-05-19_11-51-38"
SIMPLIFY      = True
DYNAMIC_BATCH = True

# ══════════════════════════════════════════════════════════════════════════════
#  PATCH ERFC, replace unsupported TF op with 1 - Erf(x)
# ══════════════════════════════════════════════════════════════════════════════

def patch_erfc(onnx_path: pathlib.Path):
    import onnx
    from onnx import helper, TensorProto

    model = onnx.load(str(onnx_path))
    new_nodes = []
    patched = 0

    for node in model.graph.node:
        if node.op_type == "Erfc":
            inp      = node.input[0]
            out      = node.output[0]
            erf_out  = out + "_erf"
            one_name = out + "_one"
            new_nodes.append(helper.make_node("Erf", [inp], [erf_out]))
            new_nodes.append(helper.make_node(
                "Constant", [], [one_name],
                value=helper.make_tensor("", TensorProto.FLOAT, [], [1.0])
            ))
            new_nodes.append(helper.make_node("Sub", [one_name, erf_out], [out]))
            patched += 1
        else:
            new_nodes.append(node)

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    onnx.save(model, str(onnx_path))
    print(f"✓ Patched {patched} Erfc node(s)")

# ══════════════════════════════════════════════════════════════════════════════
#  EXPORT
# ══════════════════════════════════════════════════════════════════════════════

def export():
    print("=" * 60)
    print("OCR Model Export → ONNX")
    print("=" * 60)

    if not KERAS_MODEL.exists():
        print(f"Model not found: {KERAS_MODEL}")
        sys.exit(1)

    print(f"Model : {KERAS_MODEL}")
    print(f"Output: {SAVE_DIR}\n")

    from fast_plate_ocr.train.model.config import load_plate_config_from_yaml
    from fast_plate_ocr.train.utilities.utils import load_keras_model
    from fast_plate_ocr.cli.export import export_onnx

    print("[1/4] Loading plate config...")
    plate_config = load_plate_config_from_yaml(PLATE_CONFIG)

    print("[2/4] Loading Keras model...")
    model = load_keras_model(KERAS_MODEL, plate_config)
    print(f"Input : {model.inputs[0].shape}")
    print(f"Output: {model.outputs[0].shape}")

    print("[3/4] Exporting to ONNX...")
    out_file = SAVE_DIR / "best.onnx"
    if out_file.exists():
        out_file.unlink()

    try:
        export_onnx(
            model=model,
            plate_config=plate_config,
            out_file=out_file,
            simplify=SIMPLIFY,
            dynamic_batch=DYNAMIC_BATCH,
            skip_validation=True,   # skip validation — we patch first then validate
            onnx_input_dtype="uint8",
            onnx_data_format="channels_last",
            opset_version=17,
        )
    except Exception as e:
        if out_file.exists():
            print(f"Export wrote file despite error: {e}")
        else:
            print(f"Export failed: {e}")
            sys.exit(1)

    print("[4/4] Patching Erfc nodes...")
    patch_erfc(out_file)

    # Validate patched model
    print("Validating patched ONNX...")
    import onnxruntime as rt
    sess = rt.InferenceSession(str(out_file))
    print("✓ ONNX valid — onnxruntime loaded successfully")

    size_kb = out_file.stat().st_size / 1024
    print(f"\n✓ Done.")
    print(f"  Saved : {out_file}")
    print(f"  Size  : {size_kb:.1f} KB")
    print(f"\nTo use:")
    print(f"from fast_plate_ocr import LicensePlateRecognizer")
    print(f"model = LicensePlateRecognizer(r'{out_file}')")

    # Cleanup tmp
    import shutil
    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR, ignore_errors=True)

if __name__ == "__main__":
    export()
