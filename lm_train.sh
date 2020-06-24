python preprocess.py \
    --only-source \
    --trainpref /mnt/my_input/wikitext-103-padded/wiki.train.tokens \
    --validpref /mnt/my_input/wikitext-103-padded/wiki.valid.tokens \
    --testpref /mnt/my_input/wikitext-103-padded/wiki.test.tokens \
    --destdir /mnt/my_output/data-bin/wikitext-103-padded \
    --workers 20

python train.py --task language_modeling \
    /mnt/my_output/data-bin/wikitext-103-padded \
    --save-dir /mnt/my_output/knn-checkpoints-padded/ \
    --arch transformer_lm_wiki103 \
    --max-update 300000 --max-lr 1.0 --t-mult 2 --lr-period-updates 270000 --lr-scheduler cosine --lr-shrink 0.75 \
    --warmup-updates 16000 --warmup-init-lr 1e-07 --min-lr 1e-09 --optimizer nag --lr 0.0001 --clip-norm 0.1 \
    --criterion adaptive_loss --max-tokens 3072 --update-freq 3 --tokens-per-sample 3072 --seed 1 --fp16 \
    --sample-break-mode none --skip-invalid-size-inputs-valid-test --ddp-backend=no_c10d --dropout 0.0 \
    --activation-dropout 0.0 --adaptive-softmax-dropout 0.0 --attention-dropout 0.0 
