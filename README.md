

- [ ] Optimize the time for TTA-Grasp
- [ ] TTA loss
- [ ] Max grasp num -> check performance




### Simulation
```
cd grasp_toolkit/simulation
python test_on_sim.py --tta-cfgs economic_grasp/configs/graspnet1b/notta


CUDA_VISIBLE_DEVICES=1 python main.py --cfg configs/graspnet1b/tta-grasp/no_mixed_forward.yaml



```