
# Predicting Point Tracks from Internet Videos enables Diverse Zero-Shot Manipulation

This repo contains code for the paper Predicting Point Tracks from Internet Videos enables Diverse Zero-Shot Manipulation

![videos](./static/glimpse.gif)


## Installation

Follow the `environment.yml` file for creating conda environment and installing dependencies. 

## Training and Inference

For training the point track prediction model, run the following after changing the number of nodes / GPUs per node, batch size as needed 

```
torchrun --nnodes=1 --nproc_per_node=8 train_track_pred.py --global-batch-size=480 --data-path=<folder with data files>
```


Specify path to initial image, goal image, and checkpoint (trained model is in [this]([https:link](https://drive.google.com/drive/folders/1UMe7ojGWcs6IiALp3K-9YpcfHV0PWcKv?usp=sharing)) link). The visualization will be saved in the folder `save_tracK_pred`. 


```
python inference_track_pred.py --ckpt=<path to model> --init=<path to initial image> --goal=<path to goal image>
```

For any questions about the project, feel free to email Homanga Bharadhwaj `hbharadh@cs.cmu.edu`


## License

The code is licensed under CC-BY-NC `License.md`

## Acknowledgement

The code in this repo is based on Diffusion Transformers `https://github.com/facebookresearch/DiT` and uses open-source packages like `diffusers`, `scipy`, `opencv`, `numpy`, `pytorch`


## Citation

If you find the repository helpful, please consider citing our paper

```
@misc{bharadhwaj2024point,
                            title={Track2Act:Predicting Point Tracks from Internet Videos enables Diverse Zero-Shot Manipulation},
                            author={Homanga Bharadhwaj and Roozbeh Mottaghi and Abhinav Gupta and Shubham Tulsiani },
                            year={2024},
                            eprint={2309.01918},
                            archivePrefix={arXiv},
                            primaryClass={cs.RO}
                      }
```
