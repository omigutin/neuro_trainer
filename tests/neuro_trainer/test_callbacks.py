from neuro_trainer.callbacks import StopFlag


def test_stop_flag(tmp_path):
    flag = StopFlag(tmp_path / "run" / "stop.request")
    assert not flag.is_set()
    flag.request()
    assert flag.is_set()
    flag.reset()
    flag.reset()
    assert not flag.is_set()
