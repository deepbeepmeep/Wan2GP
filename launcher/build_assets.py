"""Generate the build-time assets for the Windows launcher.

Produces the multi-resolution application icon from the repository favicon and
the VERSIONINFO resource PyInstaller stamps into WanGP.exe. Run from CI, or by
hand before building the spec locally.
"""

import argparse
import os
import sys

ICON_SIZES = (16, 24, 32, 48, 64, 128, 256)

VERSION_INFO_TEMPLATE = """VSVersionInfo(
  ffi=FixedFileInfo(
    filevers=({tuple}),
    prodvers=({tuple}),
    mask=0x3f,
    flags=0x0,
    OS=0x40004,
    fileType=0x1,
    subtype=0x0,
    date=(0, 0)
  ),
  kids=[
    StringFileInfo([
      StringTable(
        '040904B0',
        [StringStruct('CompanyName', {publisher!r}),
         StringStruct('FileDescription', 'WanGP Launcher'),
         StringStruct('FileVersion', {version!r}),
         StringStruct('InternalName', 'WanGP'),
         StringStruct('LegalCopyright', ''),
         StringStruct('OriginalFilename', 'WanGP.exe'),
         StringStruct('ProductName', 'WanGP'),
         StringStruct('ProductVersion', {version!r})])
    ]),
    VarFileInfo([VarStruct('Translation', [1033, 1200])])
  ]
)
"""


def build_icon(source, destination):
    from PIL import Image

    image = Image.open(source).convert("RGBA")
    width, height = image.size
    if width != height:
        # Pad to a square so Windows never stretches the icon.
        side = max(width, height)
        square = Image.new("RGBA", (side, side), (0, 0, 0, 0))
        square.paste(image, ((side - width) // 2, (side - height) // 2))
        image = square

    # Pillow silently drops any requested size larger than the source, which
    # would leave the icon without its 128 and 256 px variants.
    largest = max(ICON_SIZES)
    if image.size[0] < largest:
        image = image.resize((largest, largest), Image.LANCZOS)

    image.save(destination, format="ICO", sizes=[(size, size) for size in ICON_SIZES])
    return destination


def build_version_info(version, publisher, destination):
    parts = version.split(".")
    if len(parts) != 3 or not all(part.isdigit() for part in parts):
        raise SystemExit(f"Version '{version}' must look like 1.2.3")
    version_tuple = ", ".join(parts + ["0"])
    with open(destination, "w", encoding="utf-8") as handle:
        handle.write(
            VERSION_INFO_TEMPLATE.format(tuple=version_tuple, version=version, publisher=publisher)
        )
    return destination


def main(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default="0.0.0", help="Installer version, x.y.z")
    parser.add_argument("--publisher", default="Blencia")
    parser.add_argument(
        "--icon-source",
        default=os.path.join(os.path.dirname(here), "favicon.png"),
        help="PNG used as the application icon",
    )
    parser.add_argument("--outdir", default=os.path.join(here, "assets"))
    args = parser.parse_args(argv)

    os.makedirs(args.outdir, exist_ok=True)
    icon = build_icon(args.icon_source, os.path.join(args.outdir, "wangp.ico"))
    info = build_version_info(args.version, args.publisher, os.path.join(args.outdir, "version_info.txt"))
    print(f"wrote {icon}")
    print(f"wrote {info}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
