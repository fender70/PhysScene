"""Materials: appearance material instances and the label/depth post-process materials.

Any appearance can be a material path or a parametric material instance created on the fly, and
masks come from a generated post-process material that writes the custom
stencil id as RGB bits (R = bit0, G = bit1, B = bit2).
"""
from __future__ import annotations

import hashlib
import json

import unreal

from .ue_compat import asset_tools, enum_value, log, set_prop, warn

GENERATED = "/Game/PhysSceneGenerated"
MAT_DIR = f"{GENERATED}/Materials"
STENCIL_MAT = "M_PhysScene_StencilBits"
DEPTH_MAT = "M_PhysScene_DepthMeters"

_cache: dict[str, object] = {}


def _exists(path: str) -> bool:
    return unreal.EditorAssetLibrary.does_asset_exist(path)


def load_material(spec):
    """``spec`` is a material asset path or a dict
    ``{base: path, vector_params: {name: [r,g,b(,a)]}, scalar_params: {name: v}, texture_params: {name: path}}``."""
    if spec is None:
        return None
    key = json.dumps(spec, sort_keys=True)
    if key in _cache:
        return _cache[key]
    if isinstance(spec, str):
        mat = unreal.load_asset(spec)
        if mat is None:
            warn(f"material not found: {spec}")
        _cache[key] = mat
        return mat
    name = "MI_PS_" + hashlib.sha1(key.encode()).hexdigest()[:12]
    path = f"{MAT_DIR}/{name}"
    if _exists(path):
        mic = unreal.load_asset(path)
    else:
        base = unreal.load_asset(spec["base"])
        if base is None:
            warn(f"base material not found: {spec['base']}")
            return None
        mic = asset_tools().create_asset(name, MAT_DIR, unreal.MaterialInstanceConstant, unreal.MaterialInstanceConstantFactoryNew())
        mel = unreal.MaterialEditingLibrary
        mel.set_material_instance_parent(mic, base)
        for pname, v in (spec.get("vector_params") or {}).items():
            c = list(v) + [1.0] * (4 - len(v))
            mel.set_material_instance_vector_parameter_value(mic, pname, unreal.LinearColor(*c[:4]))
        for pname, v in (spec.get("scalar_params") or {}).items():
            mel.set_material_instance_scalar_parameter_value(mic, pname, float(v))
        for pname, tex in (spec.get("texture_params") or {}).items():
            mel.set_material_instance_texture_parameter_value(mic, pname, unreal.load_asset(tex))
        mel.update_material_instance(mic)
        unreal.EditorAssetLibrary.save_loaded_asset(mic)
    _cache[key] = mic
    return mic


def _post_process_material(name: str, scene_texture: str, code: str, output_type: str):
    path = f"{MAT_DIR}/{name}"
    if _exists(path):
        return unreal.load_asset(path)
    log(f"creating post-process material {path}")
    mat = asset_tools().create_asset(name, MAT_DIR, unreal.Material, unreal.MaterialFactoryNew())
    mat.set_editor_property("material_domain", unreal.MaterialDomain.MD_POST_PROCESS)
    try:
        loc = enum_value(unreal.BlendableLocation, "BL_SCENE_COLOR_AFTER_TONEMAPPING", "BL_AFTER_TONEMAPPING")
        set_prop(mat, "blendable_location", loc)
    except AttributeError:
        pass
    mel = unreal.MaterialEditingLibrary
    st = mel.create_material_expression(mat, unreal.MaterialExpressionSceneTexture, -700, 0)
    st.set_editor_property("scene_texture_id", getattr(unreal.SceneTextureId, scene_texture))
    cu = mel.create_material_expression(mat, unreal.MaterialExpressionCustom, -350, 0)
    cu.set_editor_property("code", code)
    cu.set_editor_property("output_type", getattr(unreal.CustomMaterialOutputType, output_type))
    inp = unreal.CustomInput()
    inp.set_editor_property("input_name", "s")
    cu.set_editor_property("inputs", [inp])
    mel.connect_material_expressions(st, "Color", cu, "s")
    mel.connect_material_property(cu, "", unreal.MaterialProperty.MP_EMISSIVE_COLOR)
    mel.recompile_material(mat)
    unreal.EditorAssetLibrary.save_loaded_asset(mat)
    return mat


def stencil_material():
    """Custom stencil id (0..7) -> RGB bits. Needs Custom Depth-Stencil enabled
    (the driver sets ``r.CustomDepth 3``)."""
    return _post_process_material(
        STENCIL_MAT,
        "PPI_CUSTOM_STENCIL",
        "int v = (int)round(s.r); return float3(v & 1, (v >> 1) & 1, (v >> 2) & 1);",
        "CMOT_FLOAT3",
    )


def depth_material():
    """Planar scene depth in metres (written to EXR)."""
    return _post_process_material(DEPTH_MAT, "PPI_SCENE_DEPTH", "return s.r / 100.0;", "CMOT_FLOAT1")
