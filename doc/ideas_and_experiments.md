### Experiment Plan

(MAD-Sim)
- [x] vggt* + vanilla 3dgs 
- [x] vggt* + vanilla 3dgs + gaussians optimization
- [x] vggt* + vanilla 3dgs + gaussians optimization (180views)
- [x] vggt* + 3dgs mcmc
- [ ] vggt* + 3dgs mcmc (180views)

notes: 
gaussians optimization - gaussians optimization 


plot the distribution of each Gaussian’s size using max (scale_0, scale_1, scale_2) and then filter out the tail of the distribution, for example by removing the last 1% of Gaussians. 

![Radius distribution](pics/radius_histogram.png)

From observations, some gaussians near object surfaces become overly large and protrude outside the object surface, and such Gaussians cannot be removed by the method that described in the PPT.

However, this optimization mainly removes floaters outside the object and helps make the rendering visually cleaner. At the same time, it tends to erode object surface. Even with tuning of the optimization thresholds, it is difficult to obtain a truly good enough anomaly-free reference model.