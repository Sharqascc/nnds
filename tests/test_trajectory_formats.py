import ast

from traffic_diffusion.train_trajectory_diffusion import parse_traj_txy


def test_parse_traj_txy_accepts_worldsample_format():
    s = (
        "WorldSample(t=0.0, x=1.0, y=2.0); "
        "WorldSample(t=0.1, x=3.0, y=4.0)"
    )
    t, xy = parse_traj_txy(s)
    assert t.tolist() == [0.0, 0.1]
    assert xy.tolist() == [[1.0, 2.0], [3.0, 4.0]]


def test_parse_traj_txy_accepts_literal_tuple_list():
    s = str([(0.0, 1.0, 2.0), (0.1, 3.0, 4.0)])
    t, xy = parse_traj_txy(s)
    assert t.tolist() == [0.0, 0.1]
    assert xy.tolist() == [[1.0, 2.0], [3.0, 4.0]]


def test_literal_tuple_list_is_ast_compatible():
    s = str([(0.0, 1.0, 2.0), (0.1, 3.0, 4.0)])
    obj = ast.literal_eval(s)
    assert obj == [(0.0, 1.0, 2.0), (0.1, 3.0, 4.0)]
