import numpy as np
import lymph.node
# actually, I should also include testing with several parent nodes...

obs_table = np.array([[[0.9, 0.2], 
                       [0.1, 0.8]]])
test_node = lymph.Node(name="test", state=0, p=0.2, obs_table=obs_table)

def test_report():
    test_node.report()

def test_trans_prob():
    test_node.state = 0
    p = test_node.p
    trans_prob = test_node.trans_prob(log=False)
    assert trans_prob[0] == 1 - p, "stay prob wrong"
    assert trans_prob[1] == p, "trans prob wrong"

    trans_prob = test_node.trans_prob(log=True)
    assert trans_prob[0] == np.log(1 - p), "log stay prob wrong"
    assert trans_prob[1] == np.log(p), "log trans prob wrong"

    test_node.state = 1
    trans_prob = test_node.trans_prob(log=False)
    assert trans_prob[0] == 0., "stay prob wrong"
    assert trans_prob[1] == 1., "trans prob wrong"

def test_obs_prob():
    for obs in np.array([[0], [1]]):
        test_node.state = 0
        assert test_node.obs_prob(log=False, observation=[obs]) == obs_table[0, obs, test_node.state]
        test_node.state = 1
        assert test_node.obs_prob(log=False, observation=[obs]) == obs_table[0, obs, test_node.state]