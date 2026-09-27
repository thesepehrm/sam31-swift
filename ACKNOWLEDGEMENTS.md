# Acknowledgements

sam31-swift stands on other people's work. This file credits it and carries the license notices
that apply to code or data derived from it.

## Meta FAIR: SAM 3 and SAM 3.1

The model architecture and the weights are Meta FAIR's
[SAM 3](https://github.com/facebookresearch/sam3) and SAM 3.1. The weights are released under the
[SAM License](https://github.com/facebookresearch/sam3/blob/main/LICENSE). This repository does not
include or redistribute them; users download them from the Hugging Face Hub and accept that license
themselves.

## mlx-vlm

The Swift modules are a line-by-line port of mlx-vlm 0.7.3's SAM 3 and SAM 3.1 implementation
(`mlx_vlm/models/sam3`, `mlx_vlm/models/sam3_1`), including its `separable_interpolate` Metal
kernel. The parity fixtures come from running mlx-vlm itself.
Source: [github.com/Blaizzy/mlx-vlm](https://github.com/Blaizzy/mlx-vlm).

```
MIT License

Copyright © 2025 Prince Canuma

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## OpenAI CLIP

`Sources/SAM31/Text/Resources/clip-vocab.json` and `clip-merges.txt` are the byte-level BPE vocabulary
and merges of CLIP, taken from the Hugging Face repository
[openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32). CLIP is
released under the MIT license ([github.com/openai/CLIP](https://github.com/openai/CLIP/blob/main/LICENSE)).

```
MIT License

Copyright (c) 2021 OpenAI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Pillow

`Sources/SAM31/Processing/ImagePreprocessor.swift` reimplements Pillow's bilinear resampling
(the two-pass, support-scaled filter of `Image.resize(..., BILINEAR)`) so that preprocessing matches
the Python reference byte for byte. No Pillow code is included, but the algorithm follows it closely,
so its notice is reproduced here. Pillow is licensed under the MIT-CMU (HPND) license.
Source: [github.com/python-pillow/Pillow](https://github.com/python-pillow/Pillow).

```
The Python Imaging Library (PIL) is

    Copyright © 1997-2011 by Secret Labs AB
    Copyright © 1995-2011 by Fredrik Lundh and contributors

Pillow is the friendly PIL fork. It is

    Copyright © 2010 by Jeffrey 'Alex' Clark and contributors

Like PIL, Pillow is licensed under the open source MIT-CMU License:

By obtaining, using, and/or copying this software and/or its associated
documentation, you agree that you have read, understood, and will comply
with the following terms and conditions:

Permission to use, copy, modify and distribute this software and its
documentation for any purpose and without fee is hereby granted,
provided that the above copyright notice appears in all copies, and that
both that copyright notice and this permission notice appear in supporting
documentation, and that the name of Secret Labs AB or the author not be
used in advertising or publicity pertaining to distribution of the software
without specific, written prior permission.

SECRET LABS AB AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS
SOFTWARE, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS.
IN NO EVENT SHALL SECRET LABS AB OR THE AUTHOR BE LIABLE FOR ANY SPECIAL,
INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
PERFORMANCE OF THIS SOFTWARE.
```

## mlx-swift

The package runs on Apple's [mlx-swift](https://github.com/ml-explore/mlx-swift) (MIT), its only
runtime dependency. It is fetched by Swift Package Manager and not vendored here.
