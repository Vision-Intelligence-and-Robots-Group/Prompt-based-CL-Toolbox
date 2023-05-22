# Prompt-based Continual Learning Toolbox
This repository contains awesome prompt-based continual learning methods:

**L2P**: Wang, Zifeng, et al. "[Learning to prompt for continual learning.](https://arxiv.org/pdf/2112.08654.pdf)" Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022.  
**DualPrompt**: Wang, Zifeng, et al. "[Dualprompt: Complementary prompting for rehearsal-free continual learning.](https://arxiv.org/pdf/2204.04799.pdf)" Computer Vision–ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part XXVI. Cham: Springer Nature Switzerland, 2022.  
**S-Prompts**: Wang, Yabin, Zhiwu Huang, and Xiaopeng Hong. "[S-Prompts Learning with Pre-trained Transformers: An Occam’s Razor for Domain Incremental Learning.](https://openreview.net/pdf?id=ZVe_WeMold)" Advances in Neural Information Processing Systems.

# Environment
Ubuntu 20.04.1 LTS  
NVIDIA GeForce RTX 3090  
Python 3.8

# Run
```
bash run.sh
```
You can also start with:
```
python -m torch.distributed.launch \
    --nproc_per_node <customize_by_yourself> \
    --master_port <customize_by_yourself> \
    --use_env main.py 
        <sprompt_cddb_slip, l2p_cifar100, dualp_cifar100>
```
# Results
#### Cifar100
| Methods | Final_acc@1 | Reproduce Official Code |
| ------- | ----------- | ----------------------- |
| DualP   | 85.65       | 85.59                   |
| L2P     | 83.31       | 83.58                   |

#### CDDB
| Methods       | Final_acc@1 | Reproduce Official Code |
| -----         | ----------- | ----------------------- |
| S-Prompt_cip  | 67.81       | 68.41                   |
| S-Prompt_clip | 84.88       | 85.5                    |
  