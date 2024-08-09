import subprocess
import itertools

# 超参数网格
param_grid = {
    'dropout': [0.2, 0.35, 0.5],
    'lr': [0.0001, 0.0005, 0.005, 0.001, 0.01],
    'weight_decay': [5e-4, 1e-4],
}

# 生成所有超参数组合
keys, values = zip(*param_grid.items())
all_param_combinations = [dict(zip(keys, combination)) for combination in itertools.product(*values)]

# 遍历每组超参数
for params in all_param_combinations:
    # 构建命令行参数
    cmd_args = [
        "python", "./train_CoraML.py",
        "--dataset", "CoraFromGNNLens",
        "--abs-pe", "rw",
        "--abs-pe-dim", "3",
        "--se", "khopgnn",
        "--gnn-type", "pna3",
        "--dropout", str(params['dropout']),
        "--k-hop" , "2", 
        "--num-layers", "4",
        "--lr", str(params['lr']),
        "--weight-decay", str(params['weight_decay']),
        "--warmup", "0",
        "--outdir", "logs",
        "--epochs", "200",
        "--layer-norm" ,
    ]
    
    # 执行命令
    cmd = ' '.join(cmd_args)
    print(f"Running command: {cmd}")
    subprocess.run(cmd, shell=True, check=False)

