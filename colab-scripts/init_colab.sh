# attachces to a session named "gpu", creating one with T4 gpu if it doesn't exist
# run this on your computer
ssh -t -o HostKeyAlias=colab-gpu -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o "ProxyCommand=$(command -v colab) ssh -s gpu --gpu T4 --proxy-mode" root@localhost "tmux -CC new-session -A -s main"
