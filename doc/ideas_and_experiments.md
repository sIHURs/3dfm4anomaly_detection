### A valid 3DGS point cloud
The quality criteria for a valid 3DGS point cloud are as follows:

1. The overall geometry of the object should be correct, without noticeable distortions.
2. The area around the object should be clean, with no obvious floaters, so that there is no occlusion between the camera poses and the object.
3. The object should be well reconstructed, with clear surfaces and recognizable texture details.
4. The rendered lighting and shading should be reasonable and consistent, without introducing visible artifacts.


### Experiment Plan

(MAD-Sim)
- [x] vggt* + vanilla 3dgs 
- [x] vggt* + vanilla 3dgs + gaussians optimization
- [x] vggt* + vanilla 3dgs + gaussians optimization (180views)
- [x] vggt* + 3dgs mcmc
- [ ] vggt* + 3dgs mcmc (180views)

**notes:** 
gaussians optimization:

After training the 3DGS model, I apply a post-processing step to the resulting gs pcd. Specifically, the Gaussians that lie outside the object will be removed, following the algorithm illustrated in the PPT. In addition, I also delete Gaussians with excessively large sizes, based on the distribution of their scales. plot the distribution of each Gaussian’s size using max (scale_0, scale_1, scale_2) and then filter out the tail of the distribution, for example by removing the last 1% of Gaussians. 

![Radius distribution](pics/radius_histogram.png)

From observations, some gaussians near object surfaces become overly large and protrude outside the object surface, and such Gaussians cannot be removed by the method that described in the PPT.

However, this optimization mainly removes floaters outside the object and helps make the rendering visually cleaner. At the same time, it tends to erode object surface. Even with tuning of the optimization thresholds, it is difficult to obtain a truly good enough anomaly-free reference model.

TODO


### One open question is whether the training images should include background context.
When using training images with a pure white background, the vanilla 3DGS optimization only minimizes the loss on the training views. As a result, the Gaussians tend to explain the white background as well, which leads to artificial background Gaussians being created. These Gaussians can then cause occlusions when the object is observed from novel viewpoints.