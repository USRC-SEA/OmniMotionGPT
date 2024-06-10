
# eval.py

compute regular ood metrics: mm-dist, r precision, diversity

### launch json files:

`scripts/eval_omg_ood.launch.json` evaluates OMG (our method) out-of-distribution performace.

# generate_fid_in.py

compute the distribution of training data of SMAL motions.

### launch json files:

`generate_fid_in.launch.json`

# eval_fid_out.py

evaluate FID between the in and out distribution.

# eval_mm.py

evaluate the out-of-distribution's multi-modality 10 pairs on 100 captions

# eval_id.py

evalute r precision, fid, mm-dist, diversity for in-distribution.

# eval_id_mm.py

because evaluating mm is very different from id, so we create another file to do the task.
