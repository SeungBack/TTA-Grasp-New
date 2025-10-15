
# packed, num_objects=5, gc6d
python tools/test_sim.py --num-view 1 --object-set packed/test --scene packed --num-rounds 100 --sideview --add-noise dex --force --best \
             --type contactgraspnet --model data/models/gc6d.pth --model_config ./cfgs/DCGRASP1B/Contact_GraspNet.yaml \
             --result-path tmp --num-objects 5 && python tools/test_sim.py --num-view 1 --object-set packed/test --scene packed --num-rounds 100 --sideview --add-noise dex --force --best \
             --type contactgraspnet --model data/models/g1b.pth --model_config ./cfgs/DCGRASP1B/Contact_GraspNet.yaml \
             --result-path tmp --num-objects 5 && python tools/test_sim.py --num-view 1 --object-set pile/test --scene pile --num-rounds 100 --sideview --add-noise dex --force --best \
             --type contactgraspnet --model data/models/g1b.pth --model_config ./cfgs/DCGRASP1B/Contact_GraspNet.yaml \
             --result-path tmp --num-objects 5 && python tools/test_sim.py --num-view 1 --object-set pile/test --scene pile --num-rounds 100 --sideview --add-noise dex --force --best \
             --type contactgraspnet --model data/models/gc6b.pth --model_config ./cfgs/DCGRASP1B/Contact_GraspNet.yaml \
             --result-path tmp --num-objects 5

--sim-gui 


# ACRONYM
python tools/test_sim.py --num-view 1 --object-set packed/test --scene packed --num-rounds 100 --sideview --add-noise dex --force --best \
             --type contactgraspnet --model data/models/acronym.pth --model_config ./cfgs/ACRONYM/Contact_GraspNet.yaml \
             --result-path tmp --num-objects 5

python tools/test_sim.py --num-view 1 --object-set pile/test --scene pile --num-rounds 100 --sideview --add-noise dex --force --best \
             --type contactgraspnet --model data/models/acronym.pth --model_config ./cfgs/ACRONYM/Contact_GraspNet.yaml \
             --result-path tmp --num-objects 5

# g1b
python tools/test_sim.py --num-view 1 --object-set packed/test --scene packed --num-rounds 100 --sideview --add-noise dex --force --best \
            --type contactgraspnet --model data/models/g1b.pth --model_config ./cfgs/DCGRASP1B/Contact_GraspNet.yaml              --result-path tmp --num-objects 5 

python tools/test_sim.py --num-view 1 --object-set pile/test --scene pile --num-rounds 100 --sideview --add-noise dex --force --best \
            --type contactgraspnet --model data/models/g1b.pth --model_config ./cfgs/DCGRASP1B/Contact_GraspNet.yaml              --result-path tmp --num-objects 5 


# gc6d
python tools/test_sim.py --num-view 1 --object-set packed/test --scene packed --num-rounds 100 --sideview --add-noise dex --force --best \
            --type contactgraspnet --model data/models/gc6d.pth --model_config ./cfgs/DCGRASP1B/Contact_GraspNet.yaml              --result-path tmp --num-objects 5 

python tools/test_sim.py --num-view 1 --object-set pile/test --scene pile --num-rounds 100 --sideview --add-noise dex --force --best \
            --type contactgraspnet --model data/models/gc6d.pth --model_config ./cfgs/DCGRASP1B/Contact_GraspNet.yaml              --result-path tmp --num-objects 5 



# anygrasp
```
python tools/test_sim.py --num-view 1 --object-set packed/test --scene packed --num-rounds 100 --sideview --add-noise dex --force --best \
             --type anygrasp \
             --result-path tmp --num-objects 5

python tools/test_sim.py --num-view 1 --object-set pile/test --scene pile --num-rounds 100 --sideview --add-noise dex --force --best \
             --type anygrasp \
             --result-path tmp --num-objects 10
```


