import subprocess

cmd = [
    'parallel-wavegan-decode',
    '--checkpoint', 'D:/Bsc-Thesis/stylevc-main/vocoder/checkpoint-1000000steps.pkl',
    '--feats-scp', 'outputs/feats.1.scp',
    '--outdir', 'outputs'
]
subprocess.call(cmd)
