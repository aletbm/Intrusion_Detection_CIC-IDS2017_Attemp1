from fastapi.testclient import TestClient
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from deployment.serve import app

client = TestClient(app)


def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_predict():
    sample_input = {
        "flow_iat_mean": 37347.875,
        "psh_flag_count": 1,
        "fwd_packets/s": 10.040731430053711,
        "bwd_packet_length_std": 2189.77294921875,
        "init_win_bytes_forward": 8192,
        "active_min": 0.0,
        "bwd_packets/s": 3.048394203186035,
        "subflow_fwd_bytes": 3.295836925506592,
        "active_std": 0.0,
        "urg_flag_count": 0,
        "init_win_bytes_backward": 229,
        "act_data_pkt_fwd": 1.0986123085021973,
        "fwd_iat_std": 2271.93408203125,
        "bwd_packet_length_min": 0,
        "fwd_iat_total": 3903,
        "min_packet_length": 0.0,
        "total_fwd_packets": 1.3862943649291992,
        "fwd_packet_length_mean": 8.666666984558105,
        "fwd_packet_length_std": 2.4215409755706787,
        "fin_flag_count": 0,
        "bwd_iat_std": 131279.71875,
        "min_seg_size_forward": 20,
        "bwd_iat_max": 294577,
    }

    sample_input = {"features": list(sample_input.values())}

    response = client.post("/predict", json=sample_input)

    assert response.status_code == 200
    assert "prediction" in response.json()
