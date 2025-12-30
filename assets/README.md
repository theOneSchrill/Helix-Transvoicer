# Assets

Place application assets here.

## Required files:

- `app.ico` - Application icon for Windows (256x256 recommended)

## Icon requirements:

The `app.ico` file should contain multiple sizes for best Windows compatibility:
- 256x256
- 128x128
- 64x64
- 48x48
- 32x32
- 16x16

You can create an ICO file from a PNG using online converters or tools like:
- https://icoconvert.com/
- ImageMagick: `magick convert icon.png -define icon:auto-resize=256,128,64,48,32,16 app.ico`
