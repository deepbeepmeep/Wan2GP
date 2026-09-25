"""Publish the shared Ming encoder core and split companion checkpoints.

Run only after both model variants have passed local loading and generation.
The legacy complete encoders are removed from the current repository revision
only after every replacement has been uploaded and verified by remote size.
"""

import argparse
import getpass
import os
from io import BytesIO
from pathlib import Path

from huggingface_hub import CommitOperationDelete, HfApi


REPO = "DeepBeepMeep/MingImage"
NEW_FILES = tuple(
    f"{folder}/{name}_{variant}.safetensors"
    for variant in ("bf16", "int8_convrot")
    for folder, name in (
        ("BailingMM2-Ming-Image", "BailingMM2-Ming-Image-Core"),
        ("ming_image_shared", "vision_encoder"),
        ("ming_image", "conditioning"),
        ("ming_image_layer", "conditioning"),
    )
)
OLD_FILES = tuple(
    f"{folder}/{folder}_{variant}.safetensors"
    for variant in ("bf16", "int8_convrot")
    for folder in ("BailingMM2-Ming-Image", "BailingMM2-Ming-Image-Layer")
) + tuple(
    f"BailingMM2-Ming-Image-Layer/{name}"
    for name in (
        "config.json", "preprocessor_config.json", "special_tokens_map.json",
        "tokenizer.json", "tokenizer_config.json",
    )
)
CARD_SECTION = """## Ming Image checkpoints

WanGP includes Ming Image 0.1 Design for image generation and editing, and Design-Layer for decomposing a reference image into RGBA layers. Both use the same Bailing language model and vision tower. The Bailing core is in `BailingMM2-Ming-Image/` alongside its tokenizer, and the vision tower plus image projection are in `ming_image_shared/`. Each variant's connector, FFN and conditioning projections are in `ming_image/conditioning_*` or `ming_image_layer/conditioning_*`. Select matching BF16 or INT8 ConvRot files for all three encoder pieces.

The diffusion transformer single-file checkpoints remain at the repository root. Design and Design-Layer use their respective transformer and conditioning weights and share the VAE in `ming_image/`. WanGP downloads and loads the required pieces automatically. The vision tower is a separate MMGP model, so text-only generation does not need to load it to the GPU.

Upstream checkpoints: [Design](https://huggingface.co/inclusionAI/Ming-Image-0.1-Design) and [Design-Layer](https://huggingface.co/inclusionAI/Ming-Image-0.1-Design-Layer). The released code and model assets are MIT licensed.
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_root", type=Path)
    parser.add_argument("--prune-legacy", action="store_true")
    parser.add_argument("--prune-only", action="store_true", help="Verify replacements and remove old files without uploading again")
    parser.add_argument("--token-stdin", action="store_true", help="Read a write token from standard input")
    args = parser.parse_args()
    token = getpass.getpass("", stream=None).strip() if args.token_stdin else None
    root = args.checkpoint_root.resolve()
    staging = (root / ".ming_image_split_publish").resolve()
    if not staging.is_relative_to(root):
        raise ValueError("staging directory is outside checkpoint root")
    staging.mkdir(exist_ok=True)
    for name in NEW_FILES:
        source, target = root / name, staging / name
        if not source.is_file():
            raise FileNotFoundError(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if not os.path.samefile(source, target):
                raise FileExistsError(target)
        else:
            os.link(source, target)
    actual = {p.relative_to(staging).as_posix() for p in staging.rglob("*") if p.is_file()}
    actual = {name for name in actual if not name.startswith(".cache/")}
    if actual != set(NEW_FILES):
        raise ValueError(f"unexpected staging files: {actual ^ set(NEW_FILES)}")

    api = HfApi(token=token)
    identity = api.whoami()
    account = identity["name"]
    if account != REPO.split("/", 1)[0]:
        raise PermissionError(f"authenticated as {account}, expected {REPO.split('/', 1)[0]}")
    role = (identity.get("auth") or {}).get("accessToken", {}).get("role")
    if role not in ("write", "fineGrained"):
        raise PermissionError(f"Hugging Face token role is {role!r}; a write token is required")
    api.model_info(REPO)
    if not args.prune_only:
        api.upload_large_folder(REPO, staging, repo_type="model", num_workers=4, print_report_every=60)
    remote = {item.rfilename: item.size for item in api.model_info(REPO, files_metadata=True).siblings}
    for name in NEW_FILES:
        if remote.get(name) != (root / name).stat().st_size:
            raise ValueError(f"remote size mismatch for {name}")
    print(f"Verified all {len(NEW_FILES)} split checkpoints remotely", flush=True)

    if not args.prune_only:
        card_path = api.hf_hub_download(REPO, "README.md")
        card = Path(card_path).read_text(encoding="utf-8")
        marker = "## Ming Image checkpoints"
        card = card.split(marker, 1)[0].rstrip() + "\n\n" + CARD_SECTION
        api.upload_file(path_or_fileobj=BytesIO(card.encode("utf-8")), path_in_repo="README.md",
                        repo_id=REPO, repo_type="model", commit_message="Document shared Ming encoder checkpoints")

    if args.prune_legacy or args.prune_only:
        remote = {item.rfilename for item in api.model_info(REPO).siblings}
        operations = [CommitOperationDelete(path_in_repo=name) for name in OLD_FILES if name in remote]
        if operations:
            api.create_commit(repo_id=REPO, repo_type="model", operations=operations,
                              commit_message="Remove redundant complete Ming encoders")
        remote = {item.rfilename for item in api.model_info(REPO).siblings}
        if any(name in remote for name in OLD_FILES):
            raise ValueError("legacy encoder files remain in the current repository revision")
        print(f"Removed {len(operations)} redundant files from the current revision", flush=True)


if __name__ == "__main__":
    main()
