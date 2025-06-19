# polspex
Polarised X-Ray Spectroscopy simulations and analysis using Quanty

- By Dan Porter
- Diamond Light Source Ltd


### Installation
Installation is currently in developement, but the following procedure should work:

```bash
# Copy the repo
$ cd good/location
$ git clone https://github.com/DiamondLightSource/polspex.git
# Create python env
$ cd polspex
$ conda env create -f polspex.yml
$ conda activate polspex
# Build website
$ cd fontend
$ pnpm install
$ pnpm build
# If needed, copy dist from frontend/dist to backend/polspex/src/polspex and remove frontend folder
$ cp -r dist ../backend/polspex/src/polspex
# install polspex
$ cd ../backend
$ python -m pip install .

# Now run polspex!
$ polspex
```
