# attachces to a session named "gpu", creating one with T4 gpu if it doesn't exist
ssh -t -o IdentitiesOnly=yes -o "ProxyCommand=$(command -v colab) ssh -s gpu --gpu T4 --proxy-mode" root@localhost "tmux -CC new-session -A -s main"
