from distill.metrics import accuracy, macro_f1, per_class_f1


def test_accuracy():
    assert accuracy([0, 1, 2, 2], [0, 1, 1, 2]) == 0.75


def test_macro_f1_hand_computed():
    # true: [0,0,1,1,2], pred: [0,1,1,1,2]
    # class0: tp1 fp0 fn1 -> f1=2/3 ; class1: tp2 fp1 fn0 -> f1=0.8 ; class2: tp1 -> f1=1.0
    got = macro_f1([0, 1, 1, 1, 2], [0, 0, 1, 1, 2], 3)
    assert abs(got - (2 / 3 + 0.8 + 1.0) / 3) < 1e-9


def test_absent_class_excluded_but_false_prediction_counts():
    # n_classes=3 but class 2 never occurs anywhere -> macro over classes 0,1 only
    assert macro_f1([0, 1], [0, 1], 3) == 1.0
    # class 2 predicted (wrongly) though never true -> f1=0 pulls the average down
    assert macro_f1([0, 2], [0, 1], 3) < 0.5


def test_per_class_f1_length():
    assert len(per_class_f1([0], [0], 5)) == 5
