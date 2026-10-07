"""Backbone (residual) and interface activation patching on the counterfactual scenes.

One run ("shard") handles one model x one object-pair family x one noise seed and writes the
predicted action chunks for every patch condition used in Figures 3 and 4:

  * backbone patches: residual stream at the output of VLM layer l, restricted to one token group
    (exo image, wrist image, target noun, final readout token, all text), for every layer;
  * interface patches: the conditioning read by action-expert block(s) J -- all blocks, each single
    block, and each window of three consecutive blocks;

for all 12 directed donor->recipient pairs of comparisons A, B and C. Donor activations are
captured live in the model's native dtype, and all runs share the same flow-matching noise.

Before patching, every shard checks that (1) instrumentation leaves the production output
unchanged, (2) live tokenization matches the manifest, (3) donor and recipient share noise, (4) all
self-patches reproduce the clean output, and (5) patching the full interface reproduces the donor.

Usage:
  python -m vla_patching.runner --model molmoact2 --family 0 --seed 1000 --out runs/
  python -m vla_patching.runner --task 17 --out runs/          # flat index, see data.task_spec

Model locations (environment variables):
  MOLMOACT2_MODEL     Hugging Face id or local dir   (default allenai/MolmoAct2-DROID)
  MOLMOACT2_REVISION  revision to load                (default d8c1abd8a27d8e859455bbe514df2bcc617db0fb)
  PI05_CHECKPOINT     openpi pi05_droid checkpoint converted to PyTorch (required for pi05)
  MOLMOBOT_CHECKPOINT MolmoBot-DROID checkpoint dir   (required for molmobot)
  MOLMOBOT_REPO       path to the MolmoBot/MolmoBot source dir, if `olmo` is not importable
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from .data import load_manifest, sha256, task_spec

MOLMOACT2_MODEL = os.environ.get("MOLMOACT2_MODEL", "allenai/MolmoAct2-DROID")
MOLMOACT2_REVISION = os.environ.get("MOLMOACT2_REVISION", "d8c1abd8a27d8e859455bbe514df2bcc617db0fb")
JOINT_SWEEP = False  # MolmoBot interface patches keep the recipient's proprioception (as in the paper)


def save(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    tmp.replace(p)


def versions():
    out = {"python": sys.version.split()[0]}
    for mod in ("torch", "transformers", "numpy"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            pass
    return out


def run(name, fi, seed, out, smoke=False):
    import numpy as np
    import torch
    from PIL import Image
    m = load_manifest()
    assert seed in m["seeds"]
    f = m["families"][fi]
    mm = m["models"][name]
    nl = mm["n_layers"]
    joint_enabled = name == "molmobot" and JOINT_SWEEP
    dest = Path(out) / ("smoke" if smoke else "shards") / name / f"family{fi:02d}" / f"seed{seed}"
    if (dest / "status.json").exists() and json.loads((dest / "status.json").read_text()).get("state") == "complete":
        print("Already complete:", dest, flush=True)
        return
    torch.set_num_threads(4)
    start = time.time(); captures = {}; cur = {}; handles = []; obs = {}
    for ep in f["observations"]:
        o = m["observations"][str(ep)]
        assert all(sha256(p) == o["image_sha256"][k] for k, p in o["images"].items()), f"image hash mismatch {ep}"
        obs[ep] = ([np.array(Image.open(o["images"][k]).convert("RGB")) for k in ("ext", "wrist")],
                   np.asarray(o["state"], np.float32), o["prompt"])

    def eq(a, b):
        if a is None or b is None: return a is b
        if isinstance(a, (tuple, list)): return len(a) == len(b) and all(eq(x, y) for x, y in zip(a, b))
        return torch.equal(a, b)

    def clone(a):
        if a is None: return None
        if isinstance(a, (tuple, list)): return tuple(clone(x) for x in a)
        return a.detach().clone()

    def residual_hook(i):
        def hook(_mod, _args, out):
            if cur.get("bypass"): return
            x = out[0] if isinstance(out, (tuple, list)) else out
            assert x.ndim == 3 and x.shape[0] == 1
            cur["residual_fires"][i] += 1
            cap = captures[cur["target"]]
            if cur["capture"]: cap["residual"][i] = clone(x)
            elif cur["kind"] != "vlm" and i == nl - 1:
                assert eq(x, cap["residual"][i]), "Interface patch altered recipient VLM"
            if cur["kind"] == "vlm" and i in cur["layers"]:
                group = cur["group"]; donor = cur["donor"]
                dp = mm["observations"][str(donor)]["groups"][group]
                tp = mm["observations"][str(cur["target"])]["groups"][group]
                assert len(dp) == len(tp) and len(tp) > 0
                donor_x = captures[donor]["residual"][i]
                assert donor_x.shape == x.shape and donor_x.dtype == x.dtype
                patched = x.clone(); patched[:, tp, :] = donor_x[:, dp, :]
                cur["patch_fires"] += 1
                if isinstance(out, tuple): return (patched,) + out[1:]
                if isinstance(out, list): return [patched] + out[1:]
                return patched
        return hook

    def interface(context, mask, state, noise=None):
        cap = captures[cur["target"]]
        assert len(context) == nl
        if cur["capture"]:
            cap.update(context=clone(context), mask=clone(mask), state=clone(state))
            if noise is not None: cap["noise"] = clone(noise)
        else:
            if cur["kind"] != "vlm": assert eq(context, cap["context"]), "Recipient context changed before interface intervention"
            assert eq(mask, cap["mask"]) and eq(state, cap["state"])
            if noise is not None: assert eq(noise, cap["noise"]), "Unmatched noise"
        cur["original_context"] = context
        patched = list(context)
        if cur["kind"] in ("interface", "joint") and cur["layers"]:
            donor = captures[cur["donor"]]
            assert eq(mask, donor["mask"])
            if noise is not None: assert eq(noise, donor["noise"])
            for i in cur["layers"]: patched[i] = clone(donor["context"][i])
        assert all(patched[i] is context[i] for i in range(nl)
                   if cur["kind"] not in ("interface", "joint") or i not in cur["layers"])
        return patched

    print(f"Loading {name} family {fi} {f['name']}, seed {seed}; smoke={smoke}", flush=True)
    if name == "molmobot":
        if os.environ.get("MOLMOBOT_REPO"):
            sys.path.insert(0, os.environ["MOLMOBOT_REPO"])
        from olmo.models.molmobot.inference_wrapper import SynthManipMolmoInferenceWrapper
        wrapper = SynthManipMolmoInferenceWrapper(
            checkpoint_path=os.environ["MOLMOBOT_CHECKPOINT"],
            device="cuda:0", use_bfloat16=True, norm_repo_id="synthmanip", compile_model=False)
        model = wrapper.model.eval(); ae = model.action_expert
        assert model.config.action_expert_layer_mode == "per_layer" and model.config.states_mode == "cross_attn"
        assert model.config.flow_matching_num_steps == 10
        vlm_blocks = [model.get_submodule(f"transformer.blocks.{i}") for i in range(nl)]
        action_blocks = ae.blocks

        def expert_hook(_mod, args, kwargs):
            if cur.get("bypass"): return
            noise, timestep, context = args
            if cur["interface_fires"] == 0:
                cur["patched_context"] = interface(context, kwargs["encoder_attention_mask"], kwargs["state_embeddings"], noise)
            else:
                assert context is cur["original_context"]
                assert eq(kwargs["state_embeddings"], captures[cur["target"]]["state"])
            cur["interface_fires"] += 1
            return (noise, timestep, cur["patched_context"]), kwargs

        def token_hook(_mod, args):
            if cur.get("bypass") or args[0].is_floating_point(): return
            expected = torch.tensor(mm["observations"][str(cur["target"])]["token_ids"], device=args[0].device)[None]
            assert eq(args[0], expected), "Live tokenization mismatch"
            cur["token_fires"] += 1

        handles.append(ae.register_forward_pre_hook(expert_hook, with_kwargs=True))
        handles.append(model.get_submodule("transformer.wte").register_forward_pre_hook(token_hook))

        def block_hook(i):
            def hook(_mod, args, kwargs):
                if cur.get("bypass"): return
                x, timestep, context = args
                cap = captures[cur["target"]]; n = cap["context"][i].shape[1]
                assert context.shape == (1, n + 1, ae.hidden_size)
                mask = kwargs.get("attn_mask")
                if cur["capture"] and cur["block_fires"][i] == 0:
                    cap["block_context"][i] = clone(context); cap["block_mask"][i] = clone(mask)
                target = cap["block_context"][i]
                assert eq(mask, cap["block_mask"][i])
                assert eq(context[:, n:], target[:, n:]), "State changed globally before local patch"
                selected = cur["kind"] in ("interface", "joint") and i in cur["layers"]
                donor = captures[cur["donor"]]["block_context"][i] if selected else target
                if cur["kind"] != "vlm":
                    assert eq(context[:, :n], donor[:, :n] if selected else target[:, :n])
                if selected and cur["kind"] == "joint":
                    context = context.clone(); context[:, n:] = donor[:, n:]
                    assert eq(context, donor)
                    cur["state_patch_fires"][i] += 1
                cur["block_fires"][i] += 1
                return (x, timestep, context), kwargs
            return hook

        def predict(ep):
            images, state, prompt = obs[ep]
            generator = torch.Generator(device="cuda:0").manual_seed(seed)
            with torch.inference_mode():
                return np.asarray(wrapper.get_action_chunk(images=images, task_description=prompt, state=state,
                                                           generator=generator), np.float32)

    elif name == "molmoact2":
        from transformers import AutoModelForImageTextToText, AutoProcessor
        processor = AutoProcessor.from_pretrained(MOLMOACT2_MODEL, revision=MOLMOACT2_REVISION,
                                                  trust_remote_code=True, extra_special_tokens={})
        model = AutoModelForImageTextToText.from_pretrained(MOLMOACT2_MODEL, revision=MOLMOACT2_REVISION,
                                                            trust_remote_code=True, torch_dtype=torch.bfloat16).to("cuda:0").eval()
        dtype = next(model.parameters()).dtype

        def move(inputs, dev):
            return {k: (v.to(device=dev, dtype=dtype if v.is_floating_point() else v.dtype) if torch.is_tensor(v) else v)
                    for k, v in inputs.items()}
        model._move_inputs_to_device = move
        ae = model.model.action_expert
        assert not model.config.action_expert_depth_gate
        vlm_blocks = [model.get_submodule(f"model.transformer.blocks.{i}") for i in range(nl)]
        action_blocks = ae.blocks
        original_prepare = ae.prepare_context; original_flow = model.model._run_action_flow_loop

        def prepare(**kwargs):
            if cur.get("bypass"): return original_prepare(**kwargs)
            mask = kwargs["encoder_attention_mask"]; kv = kwargs["encoder_kv_states"]
            ids = torch.tensor(mm["observations"][str(cur["target"])]["token_ids"], device=mask.device)[None]
            assert eq(mask.bool(), (ids != -1) & (ids != model.config.eos_token_id))
            assert all(x.shape == (1, ids.shape[1], 1024) for pair in kv for x in pair)
            kwargs = {**kwargs, "encoder_kv_states": interface(kv, mask, None)}
            context = original_prepare(**kwargs)
            cur["context_ptrs"] = [tuple(x.data_ptr() for x in pair) for pair in context.kv_contexts]
            cur["interface_fires"] += 1
            return context

        def flow(inputs, steps):
            if cur.get("bypass"): return original_flow(inputs, steps)
            assert steps == 10
            noise = inputs.trajectory.detach(); cap = captures[cur["target"]]
            if cur["capture"]: cap["noise"] = clone(noise)
            else: assert eq(noise, cap["noise"])
            return original_flow(inputs, steps)

        def token_hook(_mod, _args, kwargs):
            if cur.get("bypass"): return
            ids = kwargs["input_ids"]
            expected = torch.tensor(mm["observations"][str(cur["target"])]["token_ids"], device=ids.device)[None]
            assert eq(ids, expected), "Live tokenization mismatch"
            cur["token_fires"] += 1

        ae.prepare_context = prepare; model.model._run_action_flow_loop = flow
        handles.append(model.model.register_forward_pre_hook(token_hook, with_kwargs=True))

        def block_hook(i):
            def hook(_mod, _args, kwargs):
                if cur.get("bypass"): return
                assert tuple(x.data_ptr() for x in kwargs["cross_kv"]) == cur["context_ptrs"][i]
                cur["block_fires"][i] += 1
            return hook

        def predict(ep):
            images, state, prompt = obs[ep]
            torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            with torch.inference_mode():
                a = model.predict_action(processor=processor, images=[Image.fromarray(x) for x in images], task=prompt,
                                         state=state, norm_tag="franka_droid", inference_action_mode="continuous",
                                         enable_depth_reasoning=False, num_steps=10, normalize_language=True,
                                         enable_cuda_graph=False).actions
            a = a.detach().float().cpu().numpy() if torch.is_tensor(a) else np.asarray(a, np.float32)
            return a[0] if a.ndim == 3 else a

    else:
        import dataclasses
        from openpi.policies import policy_config
        from openpi.training import config
        from transformers.cache_utils import DynamicCache
        cfg = config.get_config("pi05_droid")
        cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, pytorch_compile_mode=None))
        policy = policy_config.create_trained_policy(cfg, os.environ["PI05_CHECKPOINT"], pytorch_device="cuda:0")
        model = policy._model.eval(); assert model.pi05
        vlm_blocks = model.paligemma_with_expert.paligemma.language_model.layers
        action_blocks = model.paligemma_with_expert.gemma_expert.model.layers
        original_denoise = model.denoise_step; original_embed = model.embed_prefix

        def embed(images, img_masks, lang_tokens, lang_masks):
            if not cur.get("bypass"):
                o = mm["observations"][str(cur["target"])]
                assert eq(lang_tokens, torch.tensor(o["lang_ids"], device=lang_tokens.device)[None])
                assert eq(lang_masks.bool(), torch.tensor(o["lang_mask"], device=lang_masks.device, dtype=torch.bool)[None])
                cur["token_fires"] += 1
            return original_embed(images, img_masks, lang_tokens, lang_masks)

        def denoise(state, mask, cache, noise, timestep):
            if cur.get("bypass"): return original_denoise(state, mask, cache, noise, timestep)
            if cur["interface_fires"] == 0:
                original = [cache[i] for i in range(len(cache))]
                patched = interface(original, mask, state, noise)
                cur["cache"] = DynamicCache.from_legacy_cache(tuple(patched)) if cur["kind"] == "interface" and cur["layers"] else cache
                cur["original_cache"] = cache
            else: assert cache is cur["original_cache"]
            cur["interface_fires"] += 1
            return original_denoise(state, mask, cur["cache"], noise, timestep)

        model.embed_prefix = embed; model.denoise_step = denoise

        def block_hook(i):
            def hook(_mod, _args, _kwargs):
                if not cur.get("bypass"): cur["block_fires"][i] += 1
            return hook

        def predict(ep):
            images, state, prompt = obs[ep]
            example = {"observation/exterior_image_1_left": images[0], "observation/wrist_image_left": images[1],
                       "observation/joint_position": state[:7], "observation/gripper_position": state[7:8], "prompt": prompt}
            torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            with torch.inference_mode(): return np.asarray(policy.infer(example)["actions"], np.float32)

    assert len(vlm_blocks) == len(action_blocks) == nl
    handles.extend(block.register_forward_hook(residual_hook(i)) for i, block in enumerate(vlm_blocks))
    handles.extend(block.register_forward_pre_hook(block_hook(i), with_kwargs=True) for i, block in enumerate(action_blocks))

    def infer(ep, donor=None, kind="none", layers=(), group=None, capture=False, bypass=False):
        cur.clear()
        cur.update(target=ep, donor=donor, kind=kind, layers=list(layers), group=group, capture=capture, bypass=bypass,
                   interface_fires=0, residual_fires=[0] * nl, block_fires=[0] * nl, state_patch_fires=[0] * nl,
                   patch_fires=0, token_fires=0)
        if capture: captures[ep] = dict(residual={}, block_context={}, block_mask={})
        a = predict(ep)
        assert a.shape == ((16, 8) if name == "molmobot" else (15, 8)) and np.isfinite(a).all()
        if not bypass:
            assert cur["residual_fires"] == [1] * nl and cur["block_fires"] == [10] * nl
            assert cur["token_fires"] == 1 and cur["interface_fires"] == (1 if name == "molmoact2" else 10)
            assert cur["patch_fires"] == (len(layers) if kind == "vlm" else 0)
            assert cur["state_patch_fires"] == [10 if kind == "joint" and i in layers else 0 for i in range(nl)]
            if kind != "vlm": assert eq(cur["original_context"], captures[ep]["context"]), "Original context mutated"
        return a

    # Clean runs, instrumentation and self-patch controls.
    clean = {}
    for ep in f["observations"]:
        ordinary = infer(ep, bypass=True); clean[ep] = infer(ep, capture=True)
        assert np.array_equal(ordinary, clean[ep]), "Instrumentation changed production output"
    assert all(eq(captures[ep]["noise"], captures[f["observations"][0]]["noise"]) for ep in f["observations"])
    if (dest / "clean.json").exists():
        previous = json.loads((dest / "clean.json").read_text())
        assert all(np.array_equal(clean[k], np.asarray(previous[str(k)], np.float32)) for k in clean), "Resumed clean baseline changed"
    save(dest / "clean.json", {str(k): v.tolist() for k, v in clean.items()})
    validation = dict(model=name, family=fi, seed=seed, versions=versions(), production_exact=True,
                      live_tokens_exact=True, matched_noise=True, interface_recipient_vlm_exact=True,
                      local_proprio_only=name == "molmobot", self_patch={}, full_transplant={})
    for ep in f["observations"]:
        assert np.array_equal(infer(ep), clean[ep])
        for kind in (["interface", "joint"] if joint_enabled else ["interface"]):
            assert np.array_equal(infer(ep, ep, kind, range(nl)), clean[ep])
    ep = f["observations"][0]
    for layer in mm["residual_layers"]:
        assert np.array_equal(infer(ep, ep, "vlm", [layer], "alltext"), clean[ep])
    for group in m["groups"]:
        for layer in (0, nl // 2, mm["residual_layers"][-1]):
            assert np.array_equal(infer(ep, ep, "vlm", [layer], group), clean[ep])
    validation["self_patch"] = dict(interface_all=True, residual_every_layer=True, residual_every_group=True, no_patch=True)
    # Full-interface references. MolmoBot interface patches keep the recipient's proprioception, so its
    # reference is the donor's images/prompt with the recipient's state.
    references = {}
    for scenario, pairs in f["pairs"].items():
        for d, t in pairs:
            if name == "molmobot":
                hybrid = f"{d}_state_{t}"; obs[hybrid] = (obs[d][0], obs[t][1], obs[d][2])
                ref = infer(hybrid, bypass=True)
            else: ref = clean[d]
            references[f"{d}>{t}"] = ref
            for kind in (["interface", "joint"] if joint_enabled else ["interface"]):
                patched = infer(t, d, kind, range(nl))
                expected = clean[d] if kind == "joint" else ref
                exact = bool(np.array_equal(patched, expected))
                validation["full_transplant"][f"{kind}:{d}>{t}"] = dict(
                    reference_exact=exact, reference_max_abs_error=float(np.max(np.abs(patched - expected))),
                    clean_donor_exact=bool(np.array_equal(patched, clean[d])))
                assert exact, validation["full_transplant"][f"{kind}:{d}>{t}"]
    save(dest / "references.json", {k: v.tolist() for k, v in references.items()})
    validation["joint_sweep_enabled"] = joint_enabled
    save(dest / "validation.json", validation)
    print("All production, identity, live-token, noise, and full-interface controls passed.", flush=True)

    interface_conditions = [("all", list(range(nl)))]
    interface_conditions += [(f"L{i:02d}", [i]) for i in range(nl)]
    interface_conditions += [(f"W{i:02d}_{i + 2:02d}", list(range(i, i + 3))) for i in range(nl - 2)]
    residual_layers = mm["residual_layers"]
    if smoke:
        interface_conditions = [(k, v) for k, v in interface_conditions if k in
                                {"all", "L00", f"L{nl // 2:02d}", f"L{nl - 1:02d}", "W00_02",
                                 f"W{nl // 2:02d}_{nl // 2 + 2:02d}", f"W{nl - 3:02d}_{nl - 1:02d}"}]
        residual_layers = sorted({0, nl // 2, mm["residual_layers"][-1]})
    conditions = {"vlm": [(f"{g}_L{i:02d}", [i], g) for g in m["groups"] for i in residual_layers],
                  "interface": [(k, v, None) for k, v in interface_conditions]}
    if joint_enabled: conditions["joint"] = conditions["interface"]
    total = 12 * sum(len(v) for v in conditions.values()); done = 0; patch_start = time.time()
    for scenario, pairs in f["pairs"].items():
        for d, t in pairs:
            for kind, cc in conditions.items():
                prefix = f"{scenario}_{d}_{t}_{kind}"
                file = dest / "results" / f"{prefix}.npz"; meta = dest / "results" / f"{prefix}.json"
                records = [dict(comparison=scenario, donor=d, target=t, kind=kind, condition=k, layers=ls, group=g) for k, ls, g in cc]
                if file.exists() and meta.exists():
                    existing = json.loads(meta.read_text())
                    assert existing["records"] == records
                    with np.load(file) as z:
                        assert len(z["actions"]) == len(records) and np.isfinite(z["actions"]).all()
                    done += len(cc); continue
                actions = []
                for label, layers, group in cc:
                    a = infer(t, d, kind, layers, group)
                    if kind == "vlm" and scenario == "B" and group in ("exo", "wrist") and name != "pi05":
                        # Image positions precede the instruction in MolmoAct2/MolmoBot and cannot attend to it.
                        assert np.array_equal(a, clean[t]), "Causal-backbone image B patch was not a no-op"
                    actions.append(a); done += 1
                    if done % 25 == 0 or done == total:
                        status = dict(state="running", model=name, family=fi, seed=seed, complete=done, total=total,
                                      seconds=time.time() - start, patch_seconds=time.time() - patch_start)
                        save(dest / "status.json", status)
                        print(f"{name} f{fi} seed{seed}: {done}/{total}; {status['seconds']:.1f}s", flush=True)
                file.parent.mkdir(parents=True, exist_ok=True)
                tmp = file.with_suffix(".tmp.npz"); np.savez_compressed(tmp, actions=np.stack(actions)); tmp.replace(file)
                save(meta, dict(records=records))
    save(dest / "status.json", dict(state="complete", model=name, family=fi, seed=seed, complete=done, total=total,
                                    seconds=time.time() - start, patch_seconds=time.time() - patch_start))
    for handle in handles: handle.remove()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=("molmoact2", "pi05", "molmobot"))
    ap.add_argument("--family", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--task", type=int, help="flat index 0..311 (overrides --model/--family/--seed)")
    ap.add_argument("--out", default="runs", help="output root (shards are written to OUT/shards/...)")
    ap.add_argument("--smoke", action="store_true", help="3 layers and 7 interface conditions only")
    a = ap.parse_args()
    if a.task is not None:
        name, fi, seed = task_spec(a.task)
    else:
        assert a.model and a.family is not None and a.seed is not None, "give --task or --model/--family/--seed"
        name, fi, seed = a.model, a.family, a.seed
    run(name, fi, seed, a.out, a.smoke)


if __name__ == "__main__":
    main()
