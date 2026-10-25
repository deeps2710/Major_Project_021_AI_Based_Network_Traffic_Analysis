# Dataset Information

This project uses the **CIC-IDS2017 (Canadian Institute for Cybersecurity Intrusion Detection System)** dataset.

## Dataset Files
The original dataset consists of the following 8 CSV captures:
- Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv (77 MB)
- Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv (76 MB)
- Friday-WorkingHours-Morning.pcap_ISCX.csv (58 MB)
- Monday-WorkingHours.pcap_ISCX.csv (176 MB)
- Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv (83 MB)
- Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv (52 MB)
- Tuesday-WorkingHours.pcap_ISCX.csv (135 MB)
- Wednesday-workingHours.pcap_ISCX.csv (225 MB)

## Notice on GitHub File Limits
Due to GitHub's **100 MB per-file size limit**, raw CSV files over 100 MB cannot be hosted directly in this Git repository.

## Download Instructions
You can obtain the complete CIC-IDS2017 dataset from:
- [UNB CIC-IDS2017 Official Page](https://www.unb.ca/cic/datasets/ids-2017.html)
- [Kaggle CIC-IDS2017 Mirror](https://www.kaggle.com/datasets/cicdataset/cicids2017)

Place the extracted CSV files into this data/ folder before running python train_model.py.
