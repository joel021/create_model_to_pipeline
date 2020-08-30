
my_pipeline = Pipeline([
        ('select_features',SelectFeatures(["NOTA_DE","NOTA_MF","NOTA_GO","NOTA_EM","H_AULA_PRES"])),
        ('remove_nan',RemoveNanValues()),
        ('feature_score',MedianRow(["NOTA_DE","NOTA_MF","NOTA_GO","NOTA_EM"])),
        ('merge_two_columns1',MegeTwoColumns(["NOTA_MF","NOTA_GO"],"M_E")),
        ('merge_two_columns2',MegeTwoColumns(["NOTA_DE","NOTA_EM"],"M_H")),
        ('feture_dificuldade',FeaturesDificuldade()),
        ('feature_dense',DenseModel(9,5,100,1000,"relu"))
        ('gxboost_classifier',xgb.XGBClassifier(max_depth=6))
    ])