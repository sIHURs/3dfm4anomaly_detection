python tools/anomaly_overlap.py \
    --image scripts/experiment_RAD_nonmask_vggt_birefnet/ad_result/results_mcmc_0/spraybottle2/3dgs_imgs/querysqueezed_038/gt.png \
    --anomaly_map scripts/experiment_RAD_nonmask_vggt_birefnet/ad_result/results_mcmc_0/spraybottle2/3dgs_imgs/querysqueezed_038/anomaly.png \
    --output overleaf_asset/overlay_spraybottle2_squeezed038_birefnet.png \
    --alpha 0.6 \
    --colormap turbo

python tools/anomaly_overlap.py \
    --image scripts/experiment_RAD_nonmask_vggt_birefnet/ad_result/results_mcmc_0/cup2_upright2/3dgs_imgs/querystained_002/gt.png \
    --anomaly_map scripts/experiment_RAD_nonmask_vggt_birefnet/ad_result/results_mcmc_0/cup2_upright2/3dgs_imgs/querystained_002/anomaly.png \
    --output overleaf_asset/overlay_cup2_upright2_stained002_birefnet.png \
    --alpha 0.6 \
    --colormap turbo

python tools/anomaly_overlap.py \
    --image scripts/experiment_RAD_nonmask_vggt_birefnet/ad_result/results_mcmc_0/tennisball/3dgs_imgs/queryscratched_066/gt.png \
    --anomaly_map scripts/experiment_RAD_nonmask_vggt_birefnet/ad_result/results_mcmc_0/tennisball/3dgs_imgs/queryscratched_066/anomaly.png \
    --output overleaf_asset/overlay_tennisball_scratched066_birefnet.png \
    --alpha 0.6 \
    --colormap turbo

python tools/anomaly_overlap.py \
    --image scripts/experiment_RAD_nonmask_vggt_birefnet/ad_result/results_mcmc_0/bowl_upright/3dgs_imgs/querymissing_001/gt.png \
    --anomaly_map scripts/experiment_RAD_nonmask_vggt_birefnet/ad_result/results_mcmc_0/bowl_upright/3dgs_imgs/querymissing_001/anomaly.png \
    --output overleaf_asset/overlay_bowl_upright_missing001_birefnet.png \
    --alpha 0.6 \
    --colormap turbo