dataset=/tungstenfs/scratch/shared/gchao_ggiorget/datasets/publication
savedir=/tungstenfs/scratch/shared/gchao_ggiorget/benchmarks/spotlearn

nice -n 19 python spotlearn_train.py --dataset ${dataset}/microtubule.npz --savedir ${savedir}
nice -n 19 python spotlearn_train.py --dataset ${dataset}/particle.npz --savedir ${savedir}
nice -n 19 python spotlearn_train.py --dataset ${dataset}/receptor.npz --savedir ${savedir}
nice -n 19 python spotlearn_train.py --dataset ${dataset}/smfish.npz --savedir ${savedir}
nice -n 19 python spotlearn_train.py --dataset ${dataset}/suntag.npz --savedir ${savedir}
nice -n 19 python spotlearn_train.py --dataset ${dataset}/vesicle.npz --savedir ${savedir}

nice -n 19 python spotlearn_test.py --datasets ${dataset} --models ${savedir}/best
