# -------------------------- 测试代码（可验证运行） --------------------------
if __name__ == "__main__":
    from factorio.satisfactory.formula import formula, nx, RecipeGNN, torch, RecipeGNN_V2, device, RecipeGNN_V3
    base_weight={
        "水": 0, "铁矿石": 1/921, "石灰石": 1/693, "铜矿石": 1/369, "钦金矿石": 1/150, "煤": 1/423,
        "原油": 1/126, "硫磺": 1/108, "铝土矿": 1/123, "粗石英": 1/135, "铀": 1/21, "SAM物质": 1/102, "氮气": 1/120
    }
    target_weight={
        "奇点电池": 1,
        "镄燃料棒": 2
    }
    # 1. 构建示例配方图
    model = RecipeGNN(formula.g, target_item = "镄燃料棒")
    
    model.train_recipe_gnn(
        base_weight=base_weight,
        epochs=10000, lr=0.005, print_interval=100,
        lambda_target=10,
        lambda_base_max=0.01,
        lambda_other_pos=10
    )
    
    model.train_recipe_gnn(
        base_weight=base_weight,
        epochs=10000, lr=0.0001, print_interval=1000,
        lambda_target=10,
        lambda_base_max=0.01,
        lambda_other_pos=10
    )
    
    model = RecipeGNN_V2(formula.g, target_item = "镄燃料棒")
    
    model.train_recipe_gnn(
        base_weight=base_weight,
        epochs=10000, lr=0.005, print_interval=100,
        lambda_target=10,
        lambda_base_max=0.01,
        lambda_other_pos=10
    )