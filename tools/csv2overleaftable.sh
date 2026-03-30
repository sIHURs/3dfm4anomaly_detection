python tools/csv2overleaftable.py \
  --csv output/RAD_ad_eval_seed0/results_image_ROCAUC.csv \
  --cols ad_eval_vanilla,ad_eval_cd,ad_eval_mcmc_manel_k150 \
  --header-names vanilla_3dgs,cdgs,3dgsmcmc  \
  --output overleaf/rad_result_3dgs_ablation.tex