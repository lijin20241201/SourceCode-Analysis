# 随机森林分类器
# 通过集成多个决策树（即“森林”）来进行分类任务，并利用随机选取的数据子集来训练每棵树，以此来减少过
# 拟合并提升模型的泛化能力。

class RandomForestClassifier(ForestClassifier):
    _parameter_constraints: dict = {
        **ForestClassifier._parameter_constraints,
        **DecisionTreeClassifier._parameter_constraints,
        "class_weight": [
            StrOptions({"balanced_subsample", "balanced"}),
            dict,
            list,
            None,
        ],
    }
    _parameter_constraints.pop("splitter")
    def __init__(
        self,
        n_estimators=100, # 森林中决策树的数量，默认为100。
        *,
        criterion="gini",# 损失函数，默认为"gini"（基尼不纯度），也可以选择"entropy"（熵）或"log_loss"（对数损失）。
        # 树的最大深度，默认为None，表示节点将一直扩展直到所有叶子节点都是纯净的或者包含的样本数少于min_samples_split。
        max_depth=None,
        min_samples_split=2, # 分裂内部节点所需的最小样本数，默认为2。
        min_samples_leaf=1, # 叶节点上的最小样本数，默认为1。
        min_weight_fraction_leaf=0.0,  # 单个叶节点上的最小样本权重分数，默认为0.0。
        max_features="sqrt", # 在寻找最佳分割时要考虑的最大特征数，默认为"sqrt"
        max_leaf_nodes=None, # 树的最大叶子节点数，默认为None，表示没有限制。
        min_impurity_decrease=0.0,
        bootstrap=True, # 是否使用自助法采样，默认为True。
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        class_weight=None,
        ccp_alpha=0.0,
        max_samples=None,
        monotonic_cst=None,
    ):
        super().__init__(
            estimator=DecisionTreeClassifier(),
            n_estimators=n_estimators,
            estimator_params=(
                "criterion",
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "min_weight_fraction_leaf",
                "max_features",
                "max_leaf_nodes",
                "min_impurity_decrease",
                "random_state",
                "ccp_alpha",
                "monotonic_cst",
            ),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight,
            max_samples=max_samples,
        )

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.monotonic_cst = monotonic_cst
        self.ccp_alpha = ccp_alpha