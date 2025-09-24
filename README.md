# Dual-STF
**Dual-STF: Learning Multi-Scale Semantic–Temporal Representations via Dual-Pathway Fusion for Time Series Anomaly Detection**

This codebase provides an implementation of Dual-STF, a novel dual-path collaborative framework for multi-variable time series anomaly detection. Dual-STF integrates a frequency-enhanced temporal reconstruction path and a semantically guided structural modeling path to collaboratively perceive multimodal anomaly features, significantly enhancing detection robustness and generalization capabilities.

## 📁 Dataset

- **SMAP**, **MSL**, and **SMD** datasets were obtained from the [OmniAnomaly repository](https://github.com/NetManAIOps/OmniAnomaly).
- **SWaT** dataset was obtained from the [TranAD repository](https://github.com/imperial-qore/TranAD).
- **PSM** dataset was obtained from the [DualTF repository](https://github.com/kaist-dmlab/DualTF).

Before running the code, please make sure the datasets are organized as expected under the corresponding directory (e.g., `./data/`). For proprietary reasons, the dataset loader component is not included in the public release. In order to maintain the integrity of the interface, the relevant code interface is still retained, and users can implement the loading code themselves based on the public data and interface specifications.

---

## 🚀 Usage

```bash
## Run Single-Domain Time Series Anomaly Detection

python main.py \
    --framework Dual-STF \
    --dataset <dataset_names> \
    --win_size 100 \
    --data_path ./data \
    --input_c <Number of channels> \
    --output_c <Number of channels> \
    --d_model 128 \
    --temperature 0.1 \
    --anomaly_ratio 0.2 \
    --anomaly_score_method learnable_norm_weight \

Arguments:
--dataset: specify the dataset name, e.g., SMD, SWaT, SMAP, MSL, PSM.
--input_c: Number of input channels, e.g., 25, 55.
--output_c: Number of output channels, e.g., 25, 55.


## License
This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.
































## License
This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.
