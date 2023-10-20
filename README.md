# Source Code of TransJect
TransJect is an encoder-decoder model that guarantees a theoretical bound for layer-wise distance preservation between a pair of tokens. We propose a simple alternative to dot-product attention to ensure Lipschitz continuity. This allows TransJect to learn injective mappings to transform token representations to different manifolds with similar topology and preserve Euclidean distance between every pair of tokens in subsequent layers. 

## How to run 
Run TransJect on PTB language modelling

```
python language_model_transject.py \
	--use_ortho \
	--use_rezero \
	--n_head 4 \
	--dataset ptb \
	--d_model 512 \
	--epochs 10
```

### Citation
If you find this repo useful, please cite our paper:
```BibTex
@inproceedings{,
  author    = {Ayan Sengupta and
               Md. Shad Akhtar and
               Tanmoy Chakraborty},
  title     = {Manifold-Preserving Transformers are Effective for Short-Long Range Encoding},
  booktitle = {},
  publisher = {},
  year      = {},
  url       = {},
  doi       = {},
}
```