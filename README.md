# Image-Colorization-using-CycleGAN

SS-CycleGAN for automatic image colorization

SS-CycleGAN architecture consists of two generators (Gcolor and Ggray) and two discriminators (Dcolor and Dgray).
Gcolor translates grayscale images to color ones, while Ggray translates color images to grayscale ones.
Dcolor discriminates generated color images to ensure rationality, and Dgray does the same for generated grayscale images.
