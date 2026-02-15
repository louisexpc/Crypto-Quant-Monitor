sudo apt update
sudo apt install redis-server
sudo systemctl enable --now redis-server
redis-cli ping   # 應回 PONG
sudo systemctl status redis-server